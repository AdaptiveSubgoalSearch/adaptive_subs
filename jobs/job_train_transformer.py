import collections
import json
import functools
import logging
import math
import os
import platform
import shutil
import sys
import time
import warnings

import gin
import numpy as np
import torch
import transformers
from transformers.models.mbart import MBartForConditionalGeneration
from transformers.models.mbart.configuration_mbart import MBartConfig

from jobs.core import Job
from metric_logging import log_scalar, log_text
from utils import hf as hf_utils
from utils import hf_generate
from utils import metrics


DatasetKit = collections.namedtuple('DatasetKit', [
    'train_dataset',
    'per_epoch_eval_dataset',
    'per_iteration_eval_datasets',
])


def make_config_dict(config_path, overrides_dict):
    config_dict = json.load(open(config_path, "r"))
    for key, value in overrides_dict.items():
        if key not in config_dict:
            warnings.warn(f'Key {key} in overrides but not in config dict')
        config_dict[key] = value
    return config_dict


def verify_and_copy_checkpoint(
        checkpoint_path, target_dir,
        expected_steps_per_epoch
):
    trainer_state = json.load(
        open(os.path.join(checkpoint_path, 'trainer_state.json'))
    )
    assert trainer_state['epoch'].is_integer()
    epoch = int(trainer_state['epoch'])
    global_step = trainer_state['global_step']

    assert epoch * expected_steps_per_epoch == global_step
    # Sanity checks, that checkpoint was taken at the very end of iteration.
    assert epoch == trainer_state['num_train_epochs']
    assert global_step == trainer_state['max_steps']

    dest_path = os.path.join(target_dir, f'checkpoint-{global_step}')
    assert not os.path.exists(dest_path)
    shutil.copytree(src=checkpoint_path, dst=dest_path)

    return epoch


def compute_metrics(eval_prediction, padding_label=None):
    assert padding_label is not None

    (predictions, output_states), target_labels = eval_prediction
    del output_states
    pred_labels = np.argmax(predictions, axis=-1)
    assert target_labels.shape == pred_labels.shape
    return {
        'accuracy': metrics.compute_accuracy(target_labels, pred_labels),
        'accuracy_ignore_padding': metrics.compute_accuracy_ignore_padding(
            target_labels, pred_labels, padding_label=padding_label
        ),
        'perfect_sequence': metrics.compute_perfect_sequence(target_labels, pred_labels)
    }


class HfTrainingPipeline(Job):
    def __init__(
            self,
            tokenizer,

            model_class=transformers.MBartForConditionalGeneration,
            config_path=gin.REQUIRED,
            model_config_overrides=None,

            n_iterations=1000_000_000,
            epochs_per_iteration=1,
            eval_every_n_iterations=20,
            generate_every_n_iterations=20,
            n_training_samples=5,
            fresh_dataset_per_iteration=True,

            batch_size=3,
            learning_rate=0.001,
            lr_schedule=hf_utils.ConstantSchedule,
            label_smoothing_factor=0,

            fp16=False,
            resume_from_checkpoint=None,
            init_hf_seed=42,
            duplicate_hf_logs=False,

            output_dir=gin.REQUIRED,
    ):
        self.tokenizer = tokenizer
        self.model_class = model_class
        self.config_path = config_path
        self.model_config_overrides = (
            model_config_overrides or hf_utils.ModelConfigOverrides()
        )

        self.n_iterations = n_iterations
        self.epochs_per_iteration = epochs_per_iteration
        self.eval_every_n_iterations = eval_every_n_iterations
        self.generate_every_n_iterations = generate_every_n_iterations
        self.n_training_samples = n_training_samples
        self.fresh_dataset_per_iteration = fresh_dataset_per_iteration

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule()
        self.label_smoothing_factor = label_smoothing_factor

        self.fp16 = fp16
        self.resume_from_checkpoint = resume_from_checkpoint
        self.init_hf_seed = init_hf_seed
        self.duplicate_hf_logs = duplicate_hf_logs

        self.output_dir = output_dir

    def _generate_datasets_for_iteration(self, done_epochs):
        """
        Returns:
            DatasetKit to use for the current iteration.
        """
        raise NotImplementedError()

    def _generate_and_verify_datasets(self, done_epochs):
        datasets = self._generate_datasets_for_iteration(done_epochs)
        # Important sanity check - otherwise weird things happen, because steps_per_epoch
        # value doesn't make sense then.
        assert len(datasets.train_dataset) == self.n_training_samples
        return datasets

    def _evaluate_after_iteration(self, trainer, eval_datasets, done_epochs):
        for dataset, metric_key_prefix in eval_datasets:
            predictions = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
            hf_utils.log_predictions(
                dataset, predictions, tokenizer=self.tokenizer,
                log_prefix=f'{metric_key_prefix}_sample', done_epochs=done_epochs
            )

    def _generate_sample_sequences(self, model, eval_datasets, done_epochs):
        for dataset, metric_key_prefix in eval_datasets:
            sequences, scores = hf_generate.generate_sequences(
                model=model,
                inputs=[hf_generate.GenerationInput(
                    input_ids=dataset[0]['input_ids'],
                    attention_mask=dataset[0]['attention_mask'],
                )],
                num_return_sequences=1,
                num_beams=5,
                max_length=512,
            )
            hf_utils.log_prediction_triple(
                input_ids=dataset[0]['input_ids'],
                target_ids=dataset[0]['labels'],
                predictions_ids=sequences[0],
                tokenizer=self.tokenizer,
                log_prefix=f'generate_{metric_key_prefix}'
            )

    def execute(self):
        hf_trainer_logger = logging.getLogger('transformers.trainer')
        hf_trainer_logger.setLevel(logging.INFO)
        if self.duplicate_hf_logs:
            # This is a hack, because output of from transformer's Logger is
            # printed on stderr, but it doesn't show in stderr file in Neptune.
            # With this additional handler Neptune works.
            hf_trainer_logger.addHandler(logging.StreamHandler(stream=sys.stderr))

        log_text('host_name', platform.node())

        # Should be called before creating the model (so for example now).
        transformers.set_seed(self.init_hf_seed)

        n_gpus = torch.cuda.device_count()
        log_text('n_gpus', str(n_gpus))
        if n_gpus == 0:
            per_device_batch_size = self.batch_size
        else:
            per_device_batch_size = math.ceil(self.batch_size / n_gpus)
            divisible_batch_size = per_device_batch_size * n_gpus
            if self.batch_size != divisible_batch_size:
                warnings.warn(
                    f'Requested batch_size {self.batch_size} is not divisible '
                    f'by the number of available GPUs {n_gpus}. Increasing '
                    f'batch size to {divisible_batch_size}.'
                )
            self.batch_size = divisible_batch_size

        steps_per_epoch = math.ceil(self.n_training_samples / self.batch_size)
        steps_per_iteration = steps_per_epoch * self.epochs_per_iteration

        config_dict = make_config_dict(
            config_path=self.config_path,
            overrides_dict=self.model_config_overrides.dict
        )
        config = self.model_class.config_class(**config_dict)
        log_text('Model config', config.to_json_string(use_diff=False))

        model = self.model_class(config=config)
        log_text('Model parameters', str(model.num_parameters()))  # log_text to avoid a chart

        optimizer_fn = functools.partial(torch.optim.Adam, lr=self.learning_rate)

        done_epochs = 0
        if self.resume_from_checkpoint:
            done_epochs = verify_and_copy_checkpoint(
                checkpoint_path=self.resume_from_checkpoint,
                target_dir=os.path.join(self.output_dir, 'transformer'),
                expected_steps_per_epoch=steps_per_epoch
            )
        assert done_epochs % self.epochs_per_iteration == 0
        first_iteration = done_epochs // self.epochs_per_iteration
        log_text('first_iteration_number', str(first_iteration))
        # Ensure that data is generated with proper seed when resuming from
        # checkpoint ("train from scratch" case works as well).
        transformers.set_seed(self.init_hf_seed + first_iteration + 1)

        t_iteration = time.time()
        datasets = self._generate_and_verify_datasets(done_epochs)
        for iteration in range(first_iteration, self.n_iterations):
            t_training = time.time()

            training_args = transformers.Seq2SeqTrainingArguments(
                num_train_epochs=done_epochs + self.epochs_per_iteration,

                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,

                label_smoothing_factor=self.label_smoothing_factor,

                # By default HF evaluation accumulates outputs for all
                # evaluation batches on GPU and only after that transfers
                # them to CPU. That consumes a lot of GPU memory in case of
                # our big evaluation datasets. This parameter forces HF to
                # move outputs from GPU to CPU after every single batch.
                eval_accumulation_steps=1,

                # Watch out! trainer.__init__() method will set all random seeds
                # (of Python, numpy, torch, tf) to this value.
                # We make this seed different for each iteration, because our
                # non-HF data generation relies on global random generators.
                seed=self.init_hf_seed + iteration + 2,

                save_steps=steps_per_iteration,  # I don't see any other way to save trainer state :(
                save_total_limit=1,

                # Half precision may speed up training. There is also similar
                # option for evaluation, but metrics may be less accurate then.
                fp16=self.fp16,

                evaluation_strategy='epoch',
                output_dir=os.path.join(self.output_dir, 'transformer'),
                logging_dir=os.path.join(self.output_dir, 'tensorboard'),
                overwrite_output_dir=True,
                disable_tqdm=True,  # Disable progress bar.
                predict_with_generate=False,
            )
            assert training_args.train_batch_size == self.batch_size

            trainer = hf_utils.CustomizedSeq2SeqTrainer(
                optimizer_fn=optimizer_fn,
                lr_schedule=self.lr_schedule,

                model=model,
                args=training_args,

                train_dataset=datasets.train_dataset,
                eval_dataset=datasets.per_epoch_eval_dataset,

                tokenizer=self.tokenizer,
                data_collator=transformers.default_data_collator,

                compute_metrics=functools.partial(
                    compute_metrics, padding_label=self.tokenizer.pad_token_id
                ),
                callbacks=[
                    hf_utils.MetricLoggerCallback(),
                    hf_utils.LrLoggerCallback(),
                    # Fail fast, if trainer can't find the checkpoint.
                    hf_utils.AssertRightEpochCallback(
                        expected_epoch=done_epochs
                    ),
                ]
            )

            if iteration == 0:
                train_result = trainer.train()
            else:
                checkpoint_path = os.path.join(
                    self.output_dir,
                    f'transformer/checkpoint-{steps_per_epoch * done_epochs}'
                )
                assert os.path.exists(checkpoint_path)
                train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
            done_epochs += self.epochs_per_iteration

            # Note: variable `model` refers to model created before entering
            # the training loop. It is appartently not updated when resuming
            # from checkpoint - so `model` variable has at the most one
            # iteration of training. To access the model with the latest state,
            # use `trainer.model`.

            log_scalar('train_loss', done_epochs, train_result.training_loss)
            hf_utils.log_eval_metrics(train_result.metrics, done_epochs)

            log_scalar('time_training_iteration', done_epochs, time.time() - t_training)

            if (iteration + 1) % self.eval_every_n_iterations == 0:
                t_eval_val = time.time()
                self._evaluate_after_iteration(
                    trainer=trainer,
                    eval_datasets=datasets.per_iteration_eval_datasets,
                    done_epochs=done_epochs
                )
                log_scalar('time_eval_iteration', done_epochs, time.time() - t_eval_val)

            if (iteration + 1) % self.generate_every_n_iterations == 0:
                t_gen_seqs = time.time()
                self._generate_sample_sequences(trainer.model, datasets.per_iteration_eval_datasets, done_epochs)
                log_scalar('time_generate_seqs', done_epochs, time.time() - t_gen_seqs)

            log_scalar('time_iteration', done_epochs, time.time() - t_iteration)
            t_iteration = time.time()

            if self.fresh_dataset_per_iteration:
                datasets = self._generate_and_verify_datasets(done_epochs)
