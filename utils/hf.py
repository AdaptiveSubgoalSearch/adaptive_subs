"""Utilities for Hugging face library."""
import statistics

import gin
import numpy as np
import torch
import transformers

from metric_logging import log_scalar, log_text

_hf_models = [
    transformers.MarianMTModel,
    transformers.MBartForConditionalGeneration,
]

GENERATION_START_TOKEN_ID = {
    transformers.MarianMTModel.__name__: 1,  # pad_token_id
    transformers.MBartForConditionalGeneration.__name__: 2,  # eos_token_id
}
assert len(GENERATION_START_TOKEN_ID) == len(_hf_models)

for model in _hf_models:
    gin.external_configurable(model)


@gin.configurable
class ModelConfigOverrides:
    def __init__(self, **kwargs):
        self.dict = kwargs


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_formula_statistics(formula_pairs, done_epochs, log_prefix, threshold):
    inputs, targets = zip(*formula_pairs)
    for (sequences, name) in [(inputs, 'input'), (targets, 'target')]:
        lengths = list(map(len, sequences))
        log_scalar(
            f'{log_prefix}_{name}_length_max', done_epochs, max(lengths)
        )
        log_scalar(
            f'{log_prefix}_{name}_length_mean', done_epochs,
            statistics.mean(lengths)
        )
        log_scalar(
            f'{log_prefix}_{name}_over_{threshold}_ratio', done_epochs,
            statistics.mean([x > threshold for x in lengths])
        )


def log_eval_metrics(metrics, epoch):
    if 'epoch' in metrics:
        del metrics['epoch']
    for metric_name, value in metrics.items():
        log_scalar(metric_name, epoch, value)


class MetricLoggerCallback(transformers.TrainerCallback):
    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        super().on_evaluate(args, state, control, **kwargs)
        metrics = kwargs['metrics']
        epoch = metrics['epoch']
        log_eval_metrics(metrics, epoch)


def log_prediction_triple(
        input_ids, target_ids, predictions_ids,
        tokenizer, log_prefix
):
    """
    Args:
        input_ids: 1-dimensional list (or array or tensor) of ids
        target_ids: as above
        predictions_ids: list of lists of ids (arrays and tensors should
            work as well)
        tokenizer: tokenizer
        log_prefix: log prefix
    """
    log_text(
        f'{log_prefix}_input',
        tokenizer.decode(input_ids)
    )
    log_text(
        f'{log_prefix}_target',
        tokenizer.decode(target_ids)
    )
    for prediction_ids in predictions_ids:
        prediction_str = tokenizer.decode(prediction_ids)
        log_text(
            f'{log_prefix}_prediction', prediction_str
        )


def log_predictions(dataset, predictions, tokenizer, log_prefix, done_epochs):
    pred_token_ids = np.argmax(predictions.predictions[0][0], axis=-1)
    log_prediction_triple(
        input_ids=dataset[0]['input_ids'],
        target_ids=dataset[0]['labels'],
        predictions_ids=[pred_token_ids],
        tokenizer=tokenizer,
        log_prefix=log_prefix,
    )
    log_eval_metrics(predictions.metrics, done_epochs)


class LrLoggerCallback(transformers.TrainerCallback):
    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        super().on_step_begin(args, state, control, **kwargs)
        optimizer = kwargs['optimizer']
        for i, param_group in enumerate(optimizer.param_groups):
            learning_rate = param_group['lr']
            log_scalar(f'learning_rate_{i}', state.epoch, learning_rate)


class LrSchedule:
    """Learning rate schedule

    Class corresponding to lr_lambda parameter in
    torch.optim.lr_scheduler.LambdaLR class.
    """
    def __call__(self, past_steps):
        """Specify multiplier for learning rate for the current gradient step.

        Args:
            past_steps (int): Overall number of gradient steps done
                during the entire training so far.

        Returns:
            Multiplicative factor f. Effective learning rate will be:
            f * init_lr,
            where init_lr is learning rate passed to the optimizer's __init__ method.
        """
        raise NotImplementedError()


@gin.configurable
class ConstantSchedule(LrSchedule):
    def __call__(self, past_steps):
        return 1


@gin.configurable
class InverseSqrtWithWarmup(LrSchedule):
    def __init__(self, warmup_steps=gin.REQUIRED):
        self._warmup_steps = warmup_steps

    def __call__(self, past_steps):
        if past_steps < self._warmup_steps:
            return (past_steps + 1) / self._warmup_steps

        return (self._warmup_steps / past_steps) ** 0.5


@gin.configurable
class CustomizedSeq2SeqTrainer(transformers.Seq2SeqTrainer):
    """Hugging Face Seq2SeqTrainer adjusted to our needs.

    In HF trainers many features are compatible only with limited range of use
    cases. HF documentation encourages to override certain method for custom
    behavior. In this class we override some methods, which were incompatible
    with our use case.

    The current list of adjustments:
    * Support custom optimizer and lr_scheduler
        Original HF implementiation allows to pass custom optimizer and
        scheduler, but is doesn't work when resuming training from checkpoint,
        which is an essential part of our training pipeline.
    * Support label smoothing
        Original HF implementation supports label smoothing, but not for seq2seq
        models.
    """

    def __init__(self, optimizer_fn=gin.REQUIRED, lr_schedule=gin.REQUIRED, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_fn = optimizer_fn
        self.lr_schedule = lr_schedule

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Attributes self.optimizer and self.lr_scheduler are defined in
        # superclass.
        if self.optimizer is None:
            self.optimizer = self.optimizer_fn(self.model.parameters())

        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer, lr_lambda=self.lr_schedule,
                last_epoch=-1, verbose=True
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs["labels"]
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None:  # iff HfTrainingPipeline.label_smoothing_factor != 0
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class AssertRightEpochCallback(transformers.TrainerCallback):
    def __init__(self, expected_epoch):
        self.expected_epoch = expected_epoch

    def on_train_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        super().on_train_begin(args, state, control, **kwargs)
        assert state.epoch == self.expected_epoch, (
            f"Trainer thinks the current epoch is {state.epoch}, "
            f"but it should be {self.expected_epoch}. "
            f"Probably trainer failed to load the checkpoint - please make "
            f"sure that paths are okay."
        )
