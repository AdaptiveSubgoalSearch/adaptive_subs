"""Utilities for Hugging face library."""

import gin
import numpy as np
import torch
import transformers

from metric_logging import log_scalar, log_text


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


def log_predictions(dataset, predictions, tokenizer, log_prefix, done_epochs):
    log_text(
        f'{log_prefix}_input',
        tokenizer.decode(dataset[0]['input_ids'])
    )
    log_text(
        f'{log_prefix}_target',
        tokenizer.decode(dataset[0]['labels'])
    )
    pred_token_ids = np.argmax(predictions.predictions[0][0], axis=-1)
    log_text(
        f'{log_prefix}_prediction',
        tokenizer.decode(pred_token_ids)
    )

    output_data = np.argmax(predictions.predictions[0], axis=-1)
    output_data = [tokenizer.decode(sample) for sample in output_data]

    valid_rate, simple_valid_rate, fixed_valid_rate = tokenizer.check_validity(output_data)
    log_scalar(f'{log_prefix} valid_rate', done_epochs, valid_rate)
    log_scalar(f'{log_prefix} simple_valid_rate', done_epochs, simple_valid_rate)
    log_scalar(f'{log_prefix} fixed_valid_rate', done_epochs, fixed_valid_rate)

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
class Seq2SeqTrainerWithCustomLrSchedule(transformers.Seq2SeqTrainer):
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
