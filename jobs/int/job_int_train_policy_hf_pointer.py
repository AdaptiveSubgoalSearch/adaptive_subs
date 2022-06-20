import json
import os
import statistics
import time

import jobs.job_train_transformer as hf_job
from metric_logging import log_text, log_scalar
import supervised
from supervised.int import hf_data
from supervised.int.gen_policy_data import generate_state_destination_action_data, problem_to_policy_data_points
from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.representation.action_representation_mask import PADDING_LEXEME
from supervised import ActionRepresentationPointer
from utils.hf import log_formula_statistics


class TrainHfForIntPolicyPointer(hf_job.HfTrainingPipeline):
    def __init__(
        self,
        n_proofs=5,
        max_steps_into_future=2,
        n_val_proofs=2,
        output_dir='out/policy',
        config_path='assets/hf_configs/mbart_config_policy_pointer.json',
        resume_from_checkpoint=None,
        max_length=512,
        n_samples_per_proof=None,
        **kwargs
    ):
        self.max_steps_into_future = max_steps_into_future
        self.max_length = max_length
        self.n_samples_per_proof = n_samples_per_proof
        action_representation = ActionRepresentationPointer()
        if self.n_samples_per_proof is None:
            # This is done to calculate number of pairs per proof
            example_problem = generate_problems(n_proofs=1)[0]
            n_pairs_per_proof = len(problem_to_policy_data_points(example_problem, self.max_steps_into_future, None))
        else:
            n_pairs_per_proof = self.n_samples_per_proof
        super().__init__(
            tokenizer=hf_data.IntPolicyTokenizerPointer(
                model_max_length=self.max_length,
                padding_side='right',
                pad_token=PADDING_LEXEME,
            ),
            n_training_samples=n_proofs * n_pairs_per_proof,
            output_dir=output_dir,
            config_path=config_path,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs
        )
        self.n_proofs = n_proofs
        self.n_val_proofs = n_val_proofs
        self.max_seq_length = self.max_length

        self.representation = supervised.int.ActionRepresentationMask
        self.max_steps_into_future = max_steps_into_future

    def _generate_dataset(self, n_proofs, done_epochs, log_prefix):

        formula_pairs = generate_state_destination_action_data(
            n_proofs, self.max_steps_into_future, self.n_samples_per_proof
        )
        log_formula_statistics(
            formula_pairs, done_epochs, log_prefix,
            threshold=self.max_seq_length
        )
        return hf_data.GoalDataset.from_formula_pairs(
            formula_pairs, self.tokenizer, max_length=self.max_seq_length
        )

    def _generate_datasets_for_iteration(self, done_epochs):
        t_train_dataset = time.time()
        train_dataset = self._generate_dataset(
            self.n_proofs, done_epochs=done_epochs, log_prefix='train'
        )
        log_scalar('time_train_dataset', done_epochs, time.time() - t_train_dataset)

        t_val_dataset = time.time()
        val_dataset = self._generate_dataset(
            self.n_val_proofs, done_epochs=done_epochs, log_prefix='val'
        )
        log_scalar('time_val_dataset', done_epochs, time.time() - t_val_dataset)

        return hf_job.DatasetKit(
            train_dataset=train_dataset,
            per_epoch_eval_dataset=hf_data.Subset(train_dataset, ratio=0.1),
            per_iteration_eval_datasets=[
                (train_dataset, 'train'),
                (val_dataset, 'val')
            ]
        )
