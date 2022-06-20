import gin
from transformers import MBartForConditionalGeneration

from metric_logging import log_text
from supervised.int import InfixRepresentation, hf_data
from supervised.int.gen_subgoal_data import generate_problems
from supervised.int.representation import infix
from utils import hf
from utils import hf_generate


class GoalGeneratorINT:
    def __init__(
        self,
        generator_checkpoint_path=gin.REQUIRED,
        n_subgoals=gin.REQUIRED,
        num_beams=gin.REQUIRED,
        temperature=None,
        device=None,
    ):
        self.generator_checkpoint_path = generator_checkpoint_path
        self.n_subgoals = n_subgoals
        self.num_beams = num_beams
        self.temperature = temperature
        self.length_penalty = 1.0  # 1.0 is the default and recommended value for beam_search
        self.device = device or hf.choose_device()
        self.model = None
        self.representation = InfixRepresentation()
        self.tokenizer = hf_data.IntGoalTokenizer(
            model_max_length=512,
            padding_side='right',
            pad_token=infix.PADDING_LEXEME
        )
        self.prediction_counter = 0

    def reset_counter(self):
        self.prediction_counter = 0

    def read_counter(self):
        return self.prediction_counter

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.generator_checkpoint_path
        ).to(self.device)

    def generate_subgoals(self, current_state):
        assert self.model is not None, 'You have to construct networks before generating subgoals'
        state_formula = self.representation.proof_state_to_input_formula(current_state)
        return self.predict_subgoals(state_formula)

    def predict_subgoals(self, state_formula):
        """
        Returns:
            List of pairs: (subgoal_str, probability).
        """
        self.prediction_counter += 1
        model_inputs = self.tokenizer(
            [state_formula], max_length=512,
            padding=False, truncation=True
        )
        generation_inputs = [
            hf_generate.GenerationInput(
                input_ids=model_inputs['input_ids'][0],
                attention_mask=model_inputs['attention_mask'][0]
            )
        ]
        encoded_sequences, scores = self._generate_sequence(generation_inputs)
        raw_sequences = [
            self.tokenizer.decode(sequence)
            for sequence in encoded_sequences
        ]
        clean_sequences = list(map(self._clean_result, raw_sequences))
        probs = [None] * len(clean_sequences)
        if scores is not None:
            sequences_lengths = [
                len(sequence) + 3  # 3 for stripped initial $@ and trailing $
                for sequence in clean_sequences
            ]
            probs = hf_generate.compute_probabilities(
                scores, sequences_lengths, length_penalty=self.length_penalty
            ).tolist()
        return list(zip(clean_sequences, probs))

    def _generate_sequence(self, inputs):
        [encoded_sequences], [scores] = hf_generate.generate_sequences(
            self.model,
            inputs=inputs, num_return_sequences=self.n_subgoals,
            num_beams=self.num_beams, max_length=512,
            temperature=self.temperature,
            length_penalty=self.length_penalty,
        )
        return encoded_sequences, scores

    @staticmethod
    def _clean_result(raw_sequence):
        orig_sequence = raw_sequence
        should_log = False

        # Handle expected format: '$@formula$_______'
        sequence = raw_sequence.rstrip('_')
        if sequence[:2] == '$@':
            sequence = sequence[2:]
        else:
            should_log = True
        if sequence[-1:] == '$':
            sequence = sequence[:-1]
        expected_sequence_len = len(sequence)

        # Handle special cases
        sequence = sequence.replace('$','').replace('@','').replace('_','').replace('|', '')
        if len(sequence) != expected_sequence_len:
            should_log = True

        if should_log:
            log_text(
                'goal_generator_int_warnings',
                f'Generated formula of weird format: {orig_sequence}'
            )

        return sequence


class SamplingGoalGeneratorINT(GoalGeneratorINT):
    def __init__(self, num_beams=None, **kwargs):
        super().__init__(num_beams=1, **kwargs)

    def _generate_sequence(self, inputs):
        [encoded_sequences] = hf_generate.sample_sequences(
            self.model,
            inputs=inputs,
            num_return_sequences=self.n_subgoals,
            max_length=512,
            temperature=self.temperature,
        )
        return encoded_sequences, None
