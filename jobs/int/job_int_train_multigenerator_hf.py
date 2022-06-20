import json
import os
import time

import jobs.job_train_transformer as hf_job
from metric_logging import log_text, log_scalar
import supervised
from supervised.int import hf_data
from supervised.int.representation.infix import PADDING_LEXEME


class TrainHfForIntMultigenerator(hf_job.HfTrainingPipeline):
    def __init__(
        self,
        n_proofs=5,
        n_val_proofs=2,
        n_pairs_per_proof=1,
        per_epoch_eval_dataset_ratio=0.05,
        output_dir='out/goal',
        config_path='assets/hf_configs/mbart_config_goal.json',
        **kwargs
    ):
        super().__init__(
            tokenizer=hf_data.IntGoalTokenizer(
                model_max_length=512,
                padding_side='right',
                pad_token=PADDING_LEXEME,
            ),
            n_training_samples=n_proofs * n_pairs_per_proof,
            output_dir=output_dir,
            config_path=config_path,
            **kwargs
        )
        self.n_proofs = n_proofs
        self.n_val_proofs = n_val_proofs
        self.per_epoch_eval_dataset_ratio = per_epoch_eval_dataset_ratio
        self.max_seq_length = 512

        # There are some hacks hardcoded in InfixRepresentation,
        # but in PrefixRepresentation not.
        self.representation = supervised.int.InfixRepresentation
        combo_path = './assets/int/benchmark/field/'
        self.kl_dict = json.load(open(os.path.join(combo_path, "orders.json"), "r"))

    def _generate_dataset(self, n_proofs, done_epochs, log_prefix):
        return hf_data.generate_int_goal_dataset(
            n_proofs=n_proofs,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            representation=self.representation,
            kl_dict=self.kl_dict,
            done_epochs=done_epochs,
            log_prefix=log_prefix,
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
            per_epoch_eval_dataset=hf_data.Subset(
                dataset=train_dataset,
                ratio=self.per_epoch_eval_dataset_ratio
            ),
            per_iteration_eval_datasets=[
                # (train_dataset, 'train'),
                (val_dataset, 'val')
            ]
        )
