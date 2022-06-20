import random
import string
import time

import gin
import torch
from joblib import Parallel, delayed
import numpy as np

from envs.int.theorem_prover_env import TheoremProverEnv
from metric_logging import log_text
from proof_system.all_axioms import axiom_sets
from supervised.int.int_hacking import sample_axiom_order, \
    generate_problem__single_trial
from third_party.INT.data_generation.generate_problems import generate_problem
from utils import storage


def set_rnd_seed_and_generate_problem(
    seed, proof_length, available_axioms
):

    random.seed(seed)
    n_axioms = min(proof_length, len(available_axioms))
    problem = None
    while problem is None:
        order = sample_axiom_order(proof_length, available_axioms)
        problem = generate_problem__single_trial(
            length=proof_length,
            num_axioms=n_axioms,
            backward=True,
            transform_gt=True,  # check this
            degree=0,  # suspicious
            num_order_or_combo=1,
            orders={f"k{n_axioms}l{proof_length}": [order]},
            train_test='train',
        )
    last_proof_step = compute_final_statement(problem)
    problem.append(last_proof_step)

    return problem

def compute_final_statement(problem):
    env = TheoremProverEnv()
    env.load_problem_step(problem[-1])
    ground_truth = problem[-1]['observation']['ground_truth']
    last_axiom = problem[-1]['lemma'].name
    last_input = problem[-1]['input_entities']
    _, _, _, info = env.step((last_axiom, *last_input))
    final_statement = info['final_statement']
    last_proof_step = {'observation': {'objectives': [final_statement], 'ground_truth': ground_truth}}
    return last_proof_step


@gin.configurable
def get_available_axioms(axiom_set_name="ordered_field"):
    """

    Args:
        axiom_set_name: which axioms set to use, available values:
        "field" generates equalities only
        "ordered_field": generates equalities and inequalities
    """
    return axiom_sets[axiom_set_name]


@gin.configurable
def generate_problems(
    n_proofs, n_workers=4, proof_length=5,
    load_from_path=None, seed=None,
):
    """
    Args (selected):
        load_from_path(Optional(str)): If you want to load problems from disk
            (instead of generating them on the fly), pass here path to
            the directory with problems. When we run out of disk problems,
            we switch to the ordinary mode - and generate problems on the
            fly from that point.
    """
    problems = []
    if load_from_path:
        problem_loader = getattr(generate_problems, 'problem_loader', None)
        if problem_loader is None:
            generate_problems.problem_loader = problem_loader = \
                storage.LongListLoader(load_from_path)
        # It may return less than n_proofs - when ran out of problems.
        problems.extend(problem_loader.load(n_proofs))

    n_rem_proofs = n_proofs - len(problems)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=2**63, size=n_rem_proofs)
    # seeds = np.random.randint(2 ** 63, size=n_rem_proofs)
    log_text('seeds_for_generate_problems[:5]', str(seeds[:5]))

    problems.extend(
        Parallel(n_jobs=n_workers, verbose=9)(
            delayed(set_rnd_seed_and_generate_problem)(
                seed=seed,
                proof_length=proof_length,
                available_axioms=get_available_axioms(),
            ) for seed in seeds)
    )
    assert len(problems) == n_proofs
    return problems


def generate_problems_from_kl_dict(n_proofs, kl_dict):
    # This function is deprecated, as kl_dict is less flexible to configure
    # in runtime.
    return [
        generate_problem(
            length=5,
            num_axioms=5,
            backward=True,
            transform_gt=True,  # check this
            degree=0,  # suspicious
            num_order_or_combo=None,
            orders=kl_dict,
            train_test='train',
        )
        for _ in range(n_proofs)
    ]


@gin.configurable
def extract_state_pairs(problems, subgoal_distance, n_pairs_per_proof=1):
    state_pairs = []
    for problem in problems:
        for i in range(len(problem) - 1):
            distance = np.random.choice(subgoal_distance)
            state_pairs.append((problem[i], problem[min(i + distance, len(problem)-1)], distance))

    state_pairs = random.sample(state_pairs, len(problems)*n_pairs_per_proof)
    return state_pairs


def add_distance_token(state_str, distance):
    return state_str[0] + list(string.ascii_lowercase)[distance] + state_str[1:]


def stringify_state_data(state_pairs, representation):
    return [
        (
            add_distance_token(representation.proof_state_to_input_formula(x_state), distance),
            representation.proof_state_to_target_formula(y_state)
        )
        for x_state, y_state, distance in state_pairs
    ]


def tokenize_stringified_data(formula_pairs, representation):
    return [
        tuple(
            representation.tokenize_formula(formula)
            for formula in formula_pair
        )
        for formula_pair in formula_pairs
    ]


def encode_tokenized_data_to_seq2seq_arrays(
        tokenized_pairs, token_consts
):
    n_formulas = len(tokenized_pairs)
    x_tokenized_data, y_tokenized_data = zip(*tokenized_pairs)

    # https://keras.io/examples/nlp/lstm_seq2seq/
    max_encoder_seq_length = max(len(x) for x in x_tokenized_data)
    max_decoder_seq_length = max(len(y) for y in y_tokenized_data)

    # create one hot encoded data for training
    num_tokens = token_consts.num_tokens
    encoder_input_data = np.zeros(
        (n_formulas, max_encoder_seq_length, num_tokens),
        dtype="float32"
    )
    decoder_input_data = np.zeros(
        (n_formulas, max_decoder_seq_length, num_tokens),
        dtype="float32"
    )
    decoder_target_data = np.zeros(
        (n_formulas, max_decoder_seq_length, num_tokens),
        dtype="float32"
    )

    padding_token = token_consts.padding_token
    for i, (x, y) in enumerate(zip(x_tokenized_data, y_tokenized_data)):
        for t, char in enumerate(x):
            encoder_input_data[i, t, char] = 1.0
        encoder_input_data[i, len(x):, padding_token] = 1.0
        for t, char in enumerate(y):
            decoder_input_data[i, t, char] = 1.0
            # decoder_target_data is ahead of decoder_input_data by one timestep
            # and doesn't include the start character.
            if t > 0:
                decoder_target_data[i, t - 1, char] = 1.0
        decoder_input_data[i, len(y):, padding_token] = 1.0
        decoder_target_data[i, len(y) - 1:, padding_token] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data


@gin.configurable
def generate_formula_pairs(n_proofs, representation):
    t = time.time()

    problems = generate_problems(n_proofs)

    print(f'INT problem generation took {time.time() - t}')

    state_pairs = extract_state_pairs(problems)

    formula_pairs = stringify_state_data(state_pairs, representation)
    log_text('Sample input formula', formula_pairs[0][0])
    log_text('Sample target formula', formula_pairs[0][1])

    return formula_pairs


def generate_data(n_proofs, kl_dict, representation):
    formula_pairs = generate_formula_pairs(n_proofs, kl_dict, representation)
    
    tokenized_pairs = tokenize_stringified_data(formula_pairs, representation)

    return encode_tokenized_data_to_seq2seq_arrays(
        tokenized_pairs, representation.token_consts
    )
