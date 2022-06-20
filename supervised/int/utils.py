from proof_system.all_axioms import all_axioms_to_prove
from visualization import seq_parse

theorem_names = [theorem.name for theorem in list(all_axioms_to_prove.values())]
thm2index = {
    # node: torch.LongTensor([ind]).to(device)
    node: ind
    for ind, node in enumerate(theorem_names)
}
index2thm = {
    ind: node for ind, node in enumerate(theorem_names)
}


def count_objectives(proof_state):
    return len(proof_state['observation']['objectives'])


def get_objective(proof_state):
    assert count_objectives(proof_state) == 1
    return proof_state['observation']['objectives'][0]


def print_proof_state(init_state):
    print('Ground truth')
    for statement in init_state['observation']['ground_truth']:
        print(seq_parse.logic_statement_to_seq_string(statement))
    print('Objectives')
    for statement in init_state['observation']['objectives']:
        print(seq_parse.logic_statement_to_seq_string(statement))
