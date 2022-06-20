import torch

from transformers import MBartForConditionalGeneration

# from supervised.int import InfixRepresentation
# from supervised.int.gen_subgoal_data import generate_problems
# from supervised.int.hf_data import GoalDataset
# from supervised.int.representation import infix
from supervised.int.hf_data import GoalDataset
from supervised.rubik import hf_rubik_subgoal
from supervised.rubik.gen_rubik_data import encode_distance
from supervised.rubik.rubik_solver_utils import generate_problems_rubik
from utils import hf_generate


class GoalGeneratorRubik:
    def __init__(self,
                 generator_checkpoint_path=None,
                 n_subgoals=None,
                 num_beams=None,
                 temperature=None,
                 add_distance_token=None):

        self.generator_checkpoint_path = generator_checkpoint_path
        self.n_subgoals = n_subgoals
        self.num_beams = num_beams
        self.temperature = temperature
        self.model = None
        self.add_distance_token = add_distance_token
        # self.representation = InfixRepresentation()
        self.tokenizer = hf_rubik_subgoal.RubikGoalTokenizer(
            model_max_length=hf_rubik_subgoal.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_subgoal.PADDING_LEXEME
        )

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(self.generator_checkpoint_path)

    def generate_subgoals(self, current_state):
        assert self.model is not None, 'You have to construct networks before generating subgoals'
        # state_formula = self.representation.proof_state_to_input_formula(current_state)
        return self.predict_subgoals(current_state, self.n_subgoals, self.num_beams, self.temperature)


    def predict_subgoals(self, state_formula, n_subgoals, num_beams, temperature):
        # state_formula = self.representation.proof_state_to_input_formula(state)
        if self.add_distance_token is not None:
            state_formula = state_formula[0] + encode_distance(self.add_distance_token) + state_formula[1:]
            print('state formula:', state_formula)
        dataset = GoalDataset.from_state([state_formula], self.tokenizer, max_length=57)
        inputs = [
            hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                attention_mask=entry['attention_mask']
            )
            for entry in dataset
        ]
        # transformers.set_seed(seed)
        model_outputs = self.model.generate(
            input_ids=torch.tensor(
                [input.input_ids for input in inputs],
                dtype=torch.int64,
                device=self.model.device,
            ),
            decoder_start_token_id=2,  # eos_token_id
            num_beams=num_beams,
            num_return_sequences=n_subgoals,
            max_length=hf_rubik_subgoal.SEQUENCE_LENGTH + 1,
            temperature=temperature

        )
        results_raw = [self.tokenizer.decode(output) for output in model_outputs]
        def clean_result(result):
            # return result.replace('$','').replace('@','').replace('_','')
            return result.replace('_','')
        results = {clean_result(result) for result in results_raw}
        return results


class SamplingGoalGeneratorRubik:
    def __init__(self,
                 generator_checkpoint_path=None,
                 n_subgoals=None,
                 num_beams=None,
                 temperature=None):

        self.generator_checkpoint_path = generator_checkpoint_path
        self.n_subgoals = n_subgoals
        self.num_beams = num_beams
        self.temperature = temperature
        self.model = None
        # self.representation = InfixRepresentation()
        self.tokenizer = hf_rubik_subgoal.RubikGoalTokenizer(
            model_max_length=hf_rubik_subgoal.SEQUENCE_LENGTH,
            padding_side='right',
            pad_token=hf_rubik_subgoal.PADDING_LEXEME
        )

    def construct_networks(self):
        self.model = MBartForConditionalGeneration.from_pretrained(self.generator_checkpoint_path)

    def generate_subgoals(self, current_state):
        assert self.model is not None, 'You have to construct networks before generating subgoals'
        # state_formula = self.representation.proof_state_to_input_formula(current_state)
        return self.predict_subgoals(current_state, self.n_subgoals, self.num_beams, self.temperature)


    def predict_subgoals(self, state_formula, n_subgoals, num_beams, temperature):
        # state_formula = self.representation.proof_state_to_input_formula(state)
        dataset = GoalDataset.from_state([state_formula], self.tokenizer, max_length=56)
        inputs = [
            hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                attention_mask=entry['attention_mask']
            )
            for entry in dataset
        ]
        # transformers.set_seed(seed)
        # model_outputs = self.model.generate(
        #     input_ids=torch.tensor(
        #         [input.input_ids for input in inputs],
        #         dtype=torch.int64,
        #         device=self.model.device,
        #     ),
        #     decoder_start_token_id=2,  # eos_token_id
        #     num_beams=num_beams,
        #     num_return_sequences=n_subgoals,
        #     max_length=hf_rubik_subgoal.SEQUENCE_LENGTH + 1,
        #     temperature=temperature
        #
        # )
        model_outputs = hf_generate.sample_sequences(model=self.model,
                                                     inputs=inputs,
                                                     num_return_sequences=n_subgoals,
                                                     max_length=hf_rubik_subgoal.SEQUENCE_LENGTH + 1,
                                                     temperature=temperature)

        results_raw = [self.tokenizer.decode(output) for output in model_outputs[0]]
        def clean_result(result):
            # return result.replace('$','').replace('@','').replace('_','')
            return result.replace('_','')
        results = {clean_result(result) for result in results_raw}
        return results
