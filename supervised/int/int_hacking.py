""" Some functions from INT modified for our needs.

The functions here are on purpose minimal modifications of original INT
functions, without additional refactoring or stylistic changes.
"""
import random

from third_party.INT.data_generation.combos_and_orders import (
    get_combo_order_info, randomize_one_axiom_order,
    generate_order_from_combination
)
from third_party.INT.data_generation.forward2backward import forward_to_backward
from third_party.INT.data_generation.generate_problems import \
    get_a_forward_problem
from third_party.INT.data_generation.utils import (
    initialize_prover, generate_valid_steps, proof_agrees_with_specs,
    steps_valid, valid_combo
)


def generate_problem__single_trial(num_axioms, length, train_test, **kwargs):
    """
    Based on INT::generate_problem(), tries to generate problem once. The
    problem with original function is that it needs to get large set of
    combinations/orders as input, since some of them might be invalid.
    """

    avoid_objective_names = kwargs.get("avoid_objective_names", [])
    # Get combos or orders ready
    use_combos, use_orders, k_combos, kl_orders, available_indices = \
        get_combo_order_info(num_axioms, length, train_test, **kwargs)
    # Initialize the atomic entities and the proof
    atom_ents, prover = initialize_prover(**kwargs)

    done = False
    returned_steps = None
    for _ in [0]:
        axiom_order = randomize_one_axiom_order(use_combos, use_orders, k_combos, kl_orders, available_indices, length)
        forward_steps = get_a_forward_problem(atom_ents, prover, axiom_order, **kwargs)
        if forward_steps is None:
            continue
        try:
            # Convert the proof to backward and validate it
            returned_steps = generate_valid_steps(forward_to_backward(forward_steps))
        except TypeError:
            continue
        # Check if the proof generated satisfies the specifications given
        if not proof_agrees_with_specs(returned_steps, length, axiom_order, avoid_objective_names):
            continue
        done = True
        steps_valid(returned_steps)

    if not done:
        returned_steps = None
    return returned_steps


def sample_axiom_order(proof_length, available_axioms):
    # based on INT::generate_combinations_and_orders()
    order = None
    while True:
        combination = random.sample(
            list(available_axioms.keys()),
            k=min(proof_length, len(available_axioms))
        )
        if not valid_combo(combination):
            continue
        try:
            order = generate_order_from_combination(
                combination, proof_length, use_tuple=True)
        except IndexError:
            continue
        if not valid_combo(order):
            continue
        break
    return order
