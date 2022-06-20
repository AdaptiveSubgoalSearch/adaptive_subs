import math

import gin
import transformers
from transformers import MBartForConditionalGeneration

from jobs import core
from supervised.int.representation import infix
from supervised.int import hf_data
from utils import hf_generate


class JobIntGenerateGoalHf(core.Job):
    def __init__(
        self,
        model_class=transformers.MBartForConditionalGeneration,
        checkpoint_path=gin.REQUIRED,

        n_proofs=100,
        proofs_per_batch=8,
        num_return_sequences=3,
        num_beams=5,

        seed=5,
    ):
        """
        Args:
            checkpoint_path: Path to the HF Trainer checkpoint
                (directory named usually as f'checkpoint-{number_of_steps}').
            seed: Seed to use for data generation. Should be different than
                in training. So it should be in [0, 41] interval.
            Other args are quite self-explanatory.
        """
        self.model_class = model_class
        self.checkpoint_path = checkpoint_path

        self.n_proofs = n_proofs
        self.proofs_per_batch = proofs_per_batch
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams

        self.seed = seed

        self.representation = infix.InfixRepresentation

    def execute(self):
        model = self.model_class.from_pretrained(
            self.checkpoint_path
        )

        tokenizer = hf_data.IntGoalTokenizer(
            model_max_length=512,
            padding_side='right',
            pad_token=infix.PADDING_LEXEME
        )

        transformers.set_seed(self.seed)

        done_samples = 0
        for batch_idx in range(math.ceil(self.n_proofs / self.proofs_per_batch)):
            dataset = hf_data.generate_int_goal_dataset(
                n_proofs=min(
                    self.proofs_per_batch,
                    self.n_proofs - batch_idx * self.proofs_per_batch
                ),
                tokenizer=tokenizer,
                max_seq_length=512,
                done_epochs=batch_idx,
                log_prefix='data',
                representation=self.representation,
                kl_dict=None,
            )
            # For LSTM is was good to pad sequences to a similar length
            # like during the training. For transformer it probably doesn't
            # matter (because padding is masked), but we may consider that in
            # case of surprisingly low quality of generated senquences.

            prediction_sets, score_sets = hf_generate.generate_sequences(
                model=model,
                inputs=[
                    hf_generate.GenerationInput(
                        input_ids=entry['input_ids'],
                        attention_mask=entry['attention_mask']
                    )
                    for entry in dataset
                ],
                num_return_sequences=self.num_return_sequences,
                num_beams=self.num_beams,
                max_length=512,
            )

            for entry, predictions, scores in zip(
                    dataset, prediction_sets, score_sets
            ):
                input_str = tokenizer.decode(entry['input_ids'])
                target_str = tokenizer.decode(entry['labels'])
                print(done_samples)
                print(f'Input:\n{input_str}')
                print(f'Target:\n{target_str}')
                print(f'{"Score":8} | Prediction')
                for prediction, score in zip(predictions, scores):
                    prediction_str = tokenizer.decode(prediction)
                    print(f'{score.item():8.5f} | {prediction_str}')
                done_samples += 1
