from data_generation.generate_problems import generate_problem
from envs.int.theorem_prover_env import TheoremProverEnv
from visualization.seq_parse import logic_statement_to_seq_string


def generate_problem_with_final_statement(num_axioms, length, train_test, env=None, **kwargs):
    """This method generates problem altogether with its last trivial statement"""
    problem = generate_problem(num_axioms, length, train_test, **kwargs)
    if env is None:
        env = TheoremProverEnv()
    env.load_problem_step(problem[-1])
    last_axiom = problem[-1]['lemma'].name
    last_input = problem[-1]['input_entities']
    _, _, _, info = env.step((last_axiom, *last_input))
    final_statement = info['final_statement']
    last_proof_step = [{'observation': {'objectives': [final_statement]}}]
    problem.extend(last_proof_step)
    return problem