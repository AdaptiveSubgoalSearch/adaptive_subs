import random
import time


from supervised.int import InfixRepresentation, ActionRepresentationMask, InfixValueRepresentation
from supervised.int.gen_subgoal_data import tokenize_stringified_data, encode_tokenized_data_to_seq2seq_arrays, \
    generate_problems


def problem_to_value_data_points(problem_with_final_statement,
                                  n_samples_per_proof=None,
                                  ):

    value_representation = InfixValueRepresentation()
    state_value_data = []
    for step, state in enumerate(problem_with_final_statement[:-1]):

        state_formula = value_representation.proof_state_to_input_formula(state)
        target_formula = value_representation.distance_to_target_formula(len(problem_with_final_statement)-step-1)
        state_value_data.append((state_formula, target_formula))

    if n_samples_per_proof is not None:
        state_value_data  = random.sample(state_value_data, n_samples_per_proof)

    return state_value_data


def generate_value_data(n_proofs,  n_samples_per_proof):
    state_destination_action_data = []
    problems = generate_problems(n_proofs=n_proofs)

    for problem in problems:
        state_destination_action_data.extend(problem_to_value_data_points(problem, n_samples_per_proof))

    return state_destination_action_data


# def generate_policy_data(n_formulas, max_steps_into_future, n_samples_per_proof):
#     representation = ActionRepresentationMask()
#     t = time.time()
#     state_destination_action_data = generate_state_destination_action_data(n_formulas, max_steps_into_future, n_samples_per_proof)
#     print(f'INT problem generation took {time.time() - t}')
#     t0_preprocess = time.time()
#     tokenized_pairs= tokenize_stringified_data(state_destination_action_data, representation)
#     seq2seq_arrays = encode_tokenized_data_to_seq2seq_arrays(
#         tokenized_pairs, representation.token_consts
#     )
#     print(f'Preprocessing took {time.time() - t0_preprocess}')
#     return seq2seq_arrays