from supervised.rubik import gen_rubik_data
from supervised.rubik.gen_rubik_data import make_env_Rubik


def cube_to_string(cube):
    return gen_rubik_data.BOS_LEXEME + gen_rubik_data.cube_bin_to_str(cube) + gen_rubik_data.EOS_LEXEME


def make_RubikEnv():
    return make_env_Rubik(step_limit=1e10, shuffles=100, obs_type='basic')


def generate_problems_rubik(n_problems):
    problems = []
    env = make_RubikEnv()

    for _ in range(n_problems):
        obs = env.reset()
        episode = []

        for _ in range(1):
            episode.append(gen_rubik_data.BOS_LEXEME + gen_rubik_data.cube_bin_to_str(obs) + gen_rubik_data.EOS_LEXEME)
            obs, _, _, _ = env.step(env.action_space.sample())

        problems.append(episode)

    return problems


FACE_TOKENS, MOVE_TOKENS, COL_TO_ID, MOVE_TOKEN_TO_ID = gen_rubik_data.policy_encoding()


def decode_action(raw_action):
    if len(raw_action) < 3:
        print('Generated invalid move:', raw_action)
        return None

    move = raw_action[2]

    if move not in MOVE_TOKEN_TO_ID:
        print('Generated invalid move:', raw_action)
        return None

    return MOVE_TOKEN_TO_ID[move]
