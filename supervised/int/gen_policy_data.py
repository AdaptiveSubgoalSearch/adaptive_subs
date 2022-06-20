import random
import time


from supervised.int import InfixRepresentation, ActionRepresentationMask
from supervised.int.gen_subgoal_data import tokenize_stringified_data, encode_tokenized_data_to_seq2seq_arrays, \
    generate_problems
from supervised.int.generate_problem_with_final_statement import generate_problem_with_final_statement
from supervised.int.representation.action_representation_pointer import ActionRepresentationPointer
from supervised import ActionRepresentationPointer
from third_party.INT.data_generation.generate_problems import generate_problem


def problem_to_policy_data_points(problem_with_final_statement,
                                  max_steps_into_future,
                                  n_samples_per_proof=None,
                                  ):
    """Works only for single objective"""
    action_representation = ActionRepresentationPointer()
    state_destination_indices = set()
    for t_current in range(len(problem_with_final_statement)-1):
        for steps_into_future in range(1,max_steps_into_future+1):
            state_destination_indices.add(
                (t_current, min(t_current + steps_into_future, len(problem_with_final_statement)-1))
            )

    state_destination_action_data = []
    for state_idx, destination_idx in state_destination_indices:
        state_target_formula = action_representation.proof_states_to_policy_input_formula(
            current_state = problem_with_final_statement[state_idx],
            destination_state = problem_with_final_statement[destination_idx]
        )
        action = (problem_with_final_statement[state_idx]['lemma'].name,
                  *problem_with_final_statement[state_idx]['input_entities'])
        action_formula = action_representation.proof_step_and_action_to_formula(
            proof_step = problem_with_final_statement[state_idx],
            action = action
        )
        state_destination_action_data.append((state_target_formula, action_formula))
    if n_samples_per_proof is not None:
        state_destination_action_data  = random.sample(state_destination_action_data, n_samples_per_proof)

    return state_destination_action_data


def generate_state_destination_action_data(n_proofs, max_steps_into_future, n_samples_per_proof):
    state_destination_action_data = []
    problems = generate_problems(n_proofs=n_proofs)

    for problem in problems:
        state_destination_action_data.extend(problem_to_policy_data_points(problem, max_steps_into_future, n_samples_per_proof))

    return state_destination_action_data


def generate_policy_data(n_formulas, max_steps_into_future, n_samples_per_proof):
    representation = ActionRepresentationPointer()
    t = time.time()
    state_destination_action_data = generate_state_destination_action_data(n_formulas, max_steps_into_future, n_samples_per_proof)
    print(f'INT problem generation took {time.time() - t}')
    t0_preprocess = time.time()
    tokenized_pairs= tokenize_stringified_data(state_destination_action_data, representation)
    seq2seq_arrays = encode_tokenized_data_to_seq2seq_arrays(
        tokenized_pairs, representation.token_consts
    )
    print(f'Preprocessing took {time.time() - t0_preprocess}')
    return seq2seq_arrays