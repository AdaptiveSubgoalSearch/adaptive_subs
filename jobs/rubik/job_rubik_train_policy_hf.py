import json
import os
import statistics
import time
import torch
from transformers import MBartForConditionalGeneration

import jobs.job_train_transformer as hf_job
from jobs.rubik.job_train_transformer_rubik import HfTrainingPipelineRubik
from metric_logging import log_text, log_scalar
import supervised
from supervised.int import hf_data
from supervised.rubik import gen_rubik_data, hf_rubik_policy
from utils import hf as hf_utils


class TrainHfForRubikPolicy(HfTrainingPipelineRubik):
    def __init__(
        self,
        n_proofs=10000,
        n_val_proofs=2,
        n_pairs_per_proof=1,
        output_dir='out/goal',
        config_path='assets/hf_configs/mbart_config_goal.json',
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
                # min_length=hf_rubik_value.SEQUENCE_LENGTH,
                vocab_size=len(hf_rubik_policy.VOCABULARY)),
            **kwargs
        )
        self.n_proofs = n_proofs
        self.n_val_proofs = n_val_proofs
        self.max_seq_length = hf_rubik_policy.SEQUENCE_LENGTH

        # There are some hacks hardcoded in InfixRepresentation,
        # but in PrefixRepresentation not.
        self.representation = supervised.int.InfixRepresentation
        combo_path = './assets/int/benchmark/field/'
        self.kl_dict = json.load(open(os.path.join(combo_path, "orders.json"), "r"))

    def _log_validity_metrics(self, dataset, sequences, tokenizer, done_epochs, prefix, model=None):
        st_accuracy = 0

        for entry, prediction in zip(dataset, sequences):
            target_str = tokenizer.decode(entry['labels'])
            st_accuracy += 1 if target_str.replace(hf_rubik_policy.PADDING_LEXEME, '') == prediction.replace(hf_rubik_policy.PADDING_LEXEME, '') else 0

        st_accuracy /= len(sequences)

        log_scalar(f'{prefix} | statewise accuracy', done_epochs, st_accuracy)

    def _generate_dataset(self, n_proofs, done_epochs, log_prefix):
        formula_pairs = gen_rubik_data.generate_policy_learning_data(n_proofs)

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
            per_epoch_eval_dataset=train_dataset,
            per_iteration_eval_datasets=[
                (train_dataset, 'train'),
                (val_dataset, 'val'),
            ]
        )
