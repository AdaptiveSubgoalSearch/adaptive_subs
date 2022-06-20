import json
import os
import statistics
import time
import numpy as np

import joblib

import jobs.job_train_transformer as hf_job
from metric_logging import log_text, log_scalar
import supervised
from supervised.int import hf_data
from supervised.int.gen_policy_data import generate_state_destination_action_data, problem_to_policy_data_points
from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.representation.action_representation_mask import PADDING_LEXEME
from supervised import ActionRepresentationPointer
from supervised.int.representation import action_representation_pointer
from utils.hf import log_formula_statistics


class TrainHfForIntVerificator(hf_job.HfTrainingPipeline):
    def __init__(
        self,
        n_proofs=2,
        n_val_proofs=2,
        output_dir='out/policy',
        config_path='assets/hf_configs/mbart_config_policy_pointer.json',
        resume_from_checkpoint=None,
        max_length=512,
        dataset_path=None,
        **kwargs
    ):
        self.max_length = max_length

        super().__init__(
            tokenizer=hf_data.IntPolicyTokenizerPointer(
                model_max_length=self.max_length,
                padding_side='right',
                pad_token=PADDING_LEXEME,
            ),
            n_training_samples=n_proofs,
            output_dir=output_dir,
            config_path=config_path,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs
        )
        self.n_proofs = n_proofs
        self.n_val_proofs = n_val_proofs
        self.max_seq_length = self.max_length

        # # For loading dataset from a single file
        # dataset = joblib.load(dataset_path)
        # positive, negative = dataset
        # positive = self.prepare_dataset(positive, action_representation_pointer.VERIFICATOR_TOKENS[1])
        # negative = self.prepare_dataset(negative, action_representation_pointer.VERIFICATOR_TOKENS[0])
        # np.random.shuffle(positive)
        # np.random.shuffle(negative)
        # self.val_positive = positive[:n_val_proofs // 2]
        # self.val_negative = negative[:n_val_proofs // 2]
        # self.val_dataset = np.array(self.val_positive + self.val_negative)
        # self.dataset_positive = np.array(positive[n_val_proofs // 2:])
        # self.dataset_negative = np.array(negative[n_val_proofs // 2:])
        # self.val_positive = np.array(self.val_positive)
        # self.val_negative = np.array(self.val_negative)
        # print(f'Number of samples: {len(self.dataset_positive)} positive, {len(self.dataset_negative)} negative')

        # For loading dataset from a directory of chunks
        self.dataset_path = dataset_path
        self.datasets = [[], []]
        datasets = os.listdir(dataset_path)
        for chunk in datasets:
            if 'positive_chunk' in chunk:
                self.datasets[1].append(dataset_path + '/' + chunk)
            elif 'negative_chunk' in chunk:
                self.datasets[0].append(dataset_path + '/' + chunk)
            else:
                print(f'Invalid dataset chunk: {chunk}')

    def prepare_dataset(self, subgoals, output_token):
        return [(subgoal, action_representation_pointer.OUTPUT_START_LEXEME + output_token + action_representation_pointer.EOS_LEXEME)
                for subgoal in subgoals]

    def sample_training_data(self, n_data):
        # # For loading dataset from a single file
        # idx_p = np.random.choice(len(self.dataset_positive), size=n_data // 2)
        # idx_n = np.random.choice(len(self.dataset_negative), size=n_data // 2)
        # training_data = np.concatenate([self.dataset_positive[idx_p], self.dataset_negative[idx_n]], axis=0)
        # np.random.shuffle(training_data)
        # log_text(f'sample datapoint', str(training_data[0]))
        # return training_data

        # For loading dataset from a directory of chunks
        result = []
        for i in range(2):
            dataset_path = np.random.choice(self.datasets[i])
            data = joblib.load(dataset_path)
            data = np.array(self.prepare_dataset(data, action_representation_pointer.VERIFICATOR_TOKENS[i]))
            idx = np.random.choice(len(data), size=n_data // 2)
            result.append(data[idx])
            log_text(f'sample datapoint', str(result[-1][0]))

        result = np.concatenate(result, axis=0)
        np.random.shuffle(result)

        return result


    def _generate_datasets_for_iteration(self, done_epochs):
        train_dataset = hf_data.GoalDataset.from_formula_pairs(
            self.sample_training_data(self.n_proofs), self.tokenizer, max_length=self.max_seq_length
        )
        # val_dataset = hf_data.GoalDataset.from_formula_pairs(
        #     self.val_dataset, self.tokenizer, max_length=self.max_seq_length
        # )
        # val_positive = hf_data.GoalDataset.from_formula_pairs(
        #     self.val_positive, self.tokenizer, max_length=self.max_seq_length
        # )
        # val_negative = hf_data.GoalDataset.from_formula_pairs(
        #     self.val_negative, self.tokenizer, max_length=self.max_seq_length
        # )

        return hf_job.DatasetKit(
            train_dataset=train_dataset,
            per_epoch_eval_dataset=hf_data.Subset(train_dataset, ratio=0.05),
            per_iteration_eval_datasets=[
                # (val_dataset, 'val'),
                # (val_positive, 'positive_val'),
                # (val_negative, 'negative_val'),
            ]
        )
