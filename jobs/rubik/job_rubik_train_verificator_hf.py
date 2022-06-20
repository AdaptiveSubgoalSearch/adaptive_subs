import json
import os
import random
import statistics
import time

import joblib
import torch
import numpy as np
from transformers import MBartForConditionalGeneration

import jobs.job_train_transformer as hf_job
from jobs.rubik.job_train_transformer_rubik import HfTrainingPipelineRubik
from metric_logging import log_text, log_scalar
import supervised
from supervised.int import hf_data
from supervised.int.hf_data import GoalDataset
from supervised.rubik import gen_rubik_data, hf_rubik_policy
from utils import hf_generate
from utils import hf as hf_utils


class TrainHfForRubikVerificator(HfTrainingPipelineRubik):
    def __init__(
        self,
        n_proofs=10000,
        n_val_proofs=1000,
        n_pairs_per_proof=1,
        output_dir='out/goal',
        config_path='assets/hf_configs/mbart_config_goal.json',
        dataset_path=None,
        **kwargs
    ):
        super().__init__(
            tokenizer=hf_rubik_policy.RubikPolicyTokenizer(
                model_max_length=hf_rubik_policy.SEQUENCE_LENGTH,
                padding_side='right',
                pad_token=hf_rubik_policy.PADDING_LEXEME
            ),
            n_training_samples=n_proofs * n_pairs_per_proof,
            output_dir=output_dir,
            config_path=config_path,
            model_config_overrides=hf_utils.ModelConfigOverrides(
                max_length=hf_rubik_policy.SEQUENCE_LENGTH,
                max_position_embeddings=hf_rubik_policy.SEQUENCE_LENGTH,
                vocab_size=len(hf_rubik_policy.VOCABULARY)),
            **kwargs
        )
        self.n_proofs = n_proofs
        self.n_val_proofs = n_val_proofs
        self.max_seq_length = hf_rubik_policy.SEQUENCE_LENGTH

        dataset = joblib.load(dataset_path)
        positive, negative = dataset
        positive = self.prepare_dataset(positive, hf_rubik_policy.VERIFICATOR_OUTPUT_TOKENS[1])
        negative = self.prepare_dataset(negative, hf_rubik_policy.VERIFICATOR_OUTPUT_TOKENS[0])
        np.random.shuffle(positive)
        np.random.shuffle(negative)
        self.val_positive = positive[:n_val_proofs // 2]
        self.val_negative = negative[:n_val_proofs // 2]
        self.val_dataset = np.array(self.val_positive + self.val_negative)
        self.dataset_positive = np.array(positive[n_val_proofs // 2:])
        self.dataset_negative = np.array(negative[n_val_proofs // 2:])
        self.val_positive = np.array(self.val_positive)
        self.val_negative = np.array(self.val_negative)
        print(f'Number of samples: {len(self.dataset_positive)} positive, {len(self.dataset_negative)} negative')

        # There are some hacks hardcoded in InfixRepresentation,
        # but in PrefixRepresentation not.
        self.representation = supervised.int.InfixRepresentation
        combo_path = './assets/int/benchmark/field/'
        self.kl_dict = json.load(open(os.path.join(combo_path, "orders.json"), "r"))

    def _log_validity_metrics(self, dataset, sequences, tokenizer, done_epochs, prefix, model=None):
        return

    def _generate_dataset(self, n_proofs, done_epochs, log_prefix):
        return

    def prepare_dataset(self, subgoals, output_token):
        return [(subgoal, hf_rubik_policy.OUTPUT_START_LEXEME + output_token + hf_rubik_policy.EOS_LEXEME)
                for subgoal in subgoals]

    def sample_training_data(self, n_data):
        idx_p = np.random.choice(len(self.dataset_positive), size=n_data // 2)
        idx_n = np.random.choice(len(self.dataset_negative), size=n_data // 2)
        training_data = np.concatenate([self.dataset_positive[idx_p], self.dataset_negative[idx_n]], axis=0)
        np.random.shuffle(training_data)
        log_text(f'sample datapoint', str(training_data[0]))
        return training_data

    def _generate_datasets_for_iteration(self, done_epochs):
        train_dataset = hf_data.GoalDataset.from_formula_pairs(
            self.sample_training_data(self.n_proofs), self.tokenizer, max_length=self.max_seq_length
        )
        val_dataset = hf_data.GoalDataset.from_formula_pairs(
            self.val_dataset, self.tokenizer, max_length=self.max_seq_length
        )
        val_positive = hf_data.GoalDataset.from_formula_pairs(
            self.val_positive, self.tokenizer, max_length=self.max_seq_length
        )
        val_negative = hf_data.GoalDataset.from_formula_pairs(
            self.val_negative, self.tokenizer, max_length=self.max_seq_length
        )

        return hf_job.DatasetKit(
            train_dataset=train_dataset,
            per_epoch_eval_dataset=hf_data.Subset(train_dataset, ratio=0.1),
            per_iteration_eval_datasets=[
                # (train_dataset, 'train'),
                (val_dataset, 'val'),
                (val_positive, 'val_positive'),
                (val_negative, 'val_negative'),
            ]
        )
