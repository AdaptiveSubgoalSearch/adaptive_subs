import copy
import random
import sys

import gin
import gym
import joblib

import numpy as np
from rubik_solver import utils as rubik_solver_utils
from rubik_solver import Move as rubik_solver_moves



solver_type = 'Random'  # Kociemba | Beginner | Random
observation_type = 'basic'  # basic | cubelet

if observation_type == 'basic':
    observation_shape = (54, 6)
else:
    observation_shape = (20, 24)


@gin.configurable
def n_random_moves(value=gin.REQUIRED):
    return value


@gin.configurable
def subgoal_episode_skipped(value=gin.REQUIRED):
    return value


def make_env_Rubik(**kwargs):
    id = ("Rubik-" + str(kwargs) + "-v0").translate(str.maketrans('', '', " {}'<>()_:"))
    id = id.replace(',', '-')

    try:
        gym.envs.register(id=id, entry_point='gym_rubik.envs:RubikEnv', kwargs=kwargs)
        print("Registered environment with id = " + id)
    except gym.error.Error:
        print("Environment with id = " + id + " already registered. Continuing with that environment.")

    env = gym.make(id)

    return env


def cube_labels():
    return 'ywrogb'


def reverse_cube_labels():
    return {'y':0, 'w':1, 'r':2, 'o':3, 'g':4, 'b':5}


from gym_rubik.envs.converter import CubeConverter
tmp_converter = CubeConverter()


def cube_bin_to_str(binary_obs):
    if observation_type == 'cubelet':
        binary_obs = np.eye(6)[tmp_converter.convert_reduced_to_basic(binary_obs).astype(np.int)]

    ordered_faces = [binary_obs[i] for i in [0, 5, 2, 4, 3, 1]]
    aligned_faces = np.array([np.rot90(face, axes=(0, 1)) for face in ordered_faces])
    sticker_list = aligned_faces.reshape((-1, 6))
    string_obs = ''.join([cube_labels()[label] for label in np.where(sticker_list)[1]])
    return string_obs


def cube_str_to_bin(string_obs):
    assert len(string_obs) == 54

    stickers = [reverse_cube_labels()[x] for x in string_obs]
    indexes = np.eye(6)[stickers]
    faces = indexes.reshape((6, 3, 3, 6))
    aligned_faces = np.array([np.rot90(face, k=-1, axes=(0, 1)) for face in faces])
    ordered_faces = [aligned_faces[i] for i in [0, 5, 2, 4, 3, 1]]

    return np.array(ordered_faces)


def cube_str_to_state(string_obs):
    return np.argmax(cube_str_to_bin(string_obs), axis=-1)


def quarterize(moves):
    quarter_moves = []

    for move in moves:
        if move.double:
            move.double = False
            quarter_moves.append(move)
            quarter_moves.append(move)
        else:
            quarter_moves.append(move)

    return quarter_moves


action_to_move_lookup = {
    0: "U", 1: "U'", 2: "D", 3: "D'", 4: "F", 5: "F'",
    6: "B", 7: "B'", 8: "R", 9: "R'", 10: "L", 11: "L'"
}

move_to_action_lookup = {v: k for k, v in action_to_move_lookup.items()}


def move_to_action(move):
    return move_to_action_lookup[str(move)]


action_encoder = np.eye(len(action_to_move_lookup.keys()))


def action_to_one_hot(action):
    return action_encoder[action]


def reverse_move(move):
    if len(move) == 1:
        return move + "'"
    else:
        return move[:-1]


def reverse_move_number(move):
    if move % 2 == 0:
        return move + 1
    else:
        return move - 1


def random_move():
    sample = random.choice(list(action_to_move_lookup.values()))
    return rubik_solver_moves.Move(sample)


def rotate_sequence(sequence, move):
    change_move = {
        'X': {'U':'F', 'D':'B', 'L':'L', 'R':'R', 'F':'D', 'B':'U', 'X':'X', 'Y':'Z', 'Z':'YP'},
        'Y': {'U':'U', 'D':'D', 'L':'F', 'R':'B', 'F':'R', 'B':'L', 'X':'ZP', 'Y':'Y', 'Z':'X'},
        'Z': {'U':'L', 'D':'R', 'L':'D', 'R':'U', 'F':'F', 'B':'B', 'X':'Y', 'Y':'XP', 'Z':'Z'},
    }

    def raw_rotate(sequence, face):
        res = []

        for move in sequence:
            if move.face in ['X', 'Y', 'Z']:
                change = change_move[face][move.face]
                move.face = change[0]
                if 'p' in change:
                    move = move.reverse()
            else:
                move.face = change_move[face][move.face]
            res.append(move)

        return res

    if move.double:
        return raw_rotate(raw_rotate(sequence, move.face), move.face)

    if move.counterclockwise:
        return raw_rotate(raw_rotate(raw_rotate(sequence, move.face), move.face), move.face)

    return raw_rotate(sequence, move.face)


def normalize_sequence(sequence):
    # Remove X,Y,Z moves.
    res = []

    while len(sequence) > 0:
        move = sequence[0]
        sequence = sequence[1:]

        if move.face in ['X', 'Y', 'Z']:
            sequence = rotate_sequence(sequence, move)
        else:
            res.append(move)

    return res


def get_raw_solution(string_observation, solver):
    if solver == 'Random':
        return [random_move() for _ in range(n_random_moves() * 2)]
    else:
        return rubik_solver_utils.solve(string_observation, solver)


PADDING_LEXEME = '_'
EOS_LEXEME = '$'
OUTPUT_START_LEXEME = '@'
BOS_LEXEME = '?'


def encode_distance(dist):
    return cube_labels()[dist]  # Assume dist < 6, which is the case currently


@gin.configurable
def generate_subgoal_learning_data(num_data, distances=[4], add_distance_token=False, remove_repetitions=False):
    shuffles = 0 if solver_type == 'Random' else 100
    env = make_env_Rubik(step_limit=100, shuffles=shuffles, obs_type=observation_type)

    data_xy = []

    while True:
        obs = env.reset()
        episode = [cube_bin_to_str(obs) + EOS_LEXEME]
        # print(cube_bin_to_str(obs))
        # print(rubik_solver.solve(cube_bin_to_str(obs), 'Kociemba'))
        solution = [move_to_action(move) for move in quarterize(
            normalize_sequence(get_raw_solution(cube_bin_to_str(obs), solver_type)))]
        # solution = [0, 8] * 20
        # print(solution)
        for m in solution:
            obs, rew, done, info = env.step(m)
            obs_string = cube_bin_to_str(obs) + EOS_LEXEME
            if remove_repetitions and obs_string in set(episode):
                env.step(reverse_move_number(m))
                continue
            episode.append(obs_string)
            if len(episode) > n_random_moves():
                break
        # print(episode)
        # print(cube_bin_to_str(obs))
        # print(obs.reshape(54, 6))

        if solver_type == 'Random':
            episode = list(reversed(episode))

        for i in range(len(episode) - subgoal_episode_skipped()):
            dist = np.random.choice(distances)
            distance_token = encode_distance(dist) if add_distance_token else ''
            target_i = min(i + dist, len(episode) - 1)
            data_xy.append((BOS_LEXEME + distance_token + episode[i], OUTPUT_START_LEXEME + episode[target_i]))

        if len(data_xy) >= num_data:
            break

    print('Sample input formula', data_xy[0][0])
    print('Sample target formula', data_xy[0][1])

    random.shuffle(data_xy)

    return data_xy[:num_data]


def generate_value_learning_data(num_data, fixed_dist=None):
    shuffles = 0 if solver_type == 'Random' else 100
    env = make_env_Rubik(step_limit=100, shuffles=shuffles, obs_type=observation_type)

    data_xy = []

    while True:
        obs = env.reset()
        episode = [cube_bin_to_str(obs)]

        solution = [move_to_action(move) for move in quarterize(
            normalize_sequence(get_raw_solution(cube_bin_to_str(obs), solver_type)))]

        for m in solution:
            obs, rew, done, info = env.step(m)
            obs_string = cube_bin_to_str(obs)
            episode.append(obs_string)

        if solver_type == 'Random':
            episode = list(reversed(episode))

        # if fixed_dist is None:
        #     if np.random.rand() < 1/3:
        #         value = np.random.randint(1, n_random_moves() // 2)
        #     else:
        #         value = np.random.randint(n_random_moves() // 2, n_random_moves())
        # else:
        #     value = fixed_dist

        for value in range(n_random_moves()):
            state = episode[-(value + 1)]
            if value > n_random_moves() // 2 or np.random.rand() < 0.5:
                data_xy.append((BOS_LEXEME + state + EOS_LEXEME, OUTPUT_START_LEXEME + chr(65 + value) + EOS_LEXEME))

        if len(data_xy) >= num_data:
            break

    random.shuffle(data_xy)

    print('Sample input formula', data_xy[0][0])
    print('Sample target formula', data_xy[0][1])

    return data_xy[:num_data]


def policy_encoding():
    col_to_id = {'r': 0, 'g': 1, 'b': 2, 'w': 3, 'o': 4, 'y': 5}
    all_tokens = [chr(97 + i) for i in range(25)] + [chr(65 + i) for i in range(25)]
    face_tokens = np.array(all_tokens[:36]).reshape((6, 6))
    move_tokens = all_tokens[36:]
    move_token_to_id = {token: num for (num, token) in enumerate(move_tokens)}

    return face_tokens, move_tokens, col_to_id, move_token_to_id


def encode_policy_data(current_obs, target_obs, move):
    face_tokens, move_tokens, col_to_id, move_token_to_id = policy_encoding()

    state = [face_tokens[col_to_id[x], col_to_id[y]] for x, y in zip(current_obs, target_obs)]

    return BOS_LEXEME + ''.join(state) + EOS_LEXEME, OUTPUT_START_LEXEME + move_tokens[move] + EOS_LEXEME


def encode_policy_subgoal(current_obs, target_obs):
    current_obs = current_obs[1:-1]
    target_obs = target_obs[1:-1]
    face_tokens, move_tokens, col_to_id, move_token_to_id = policy_encoding()

    state = [face_tokens[col_to_id[x], col_to_id[y]] for x, y in zip(current_obs, target_obs)]

    return BOS_LEXEME + ''.join(state) + EOS_LEXEME


def generate_policy_learning_data(num_data, brutal=False):
    shuffles = 0 if solver_type == 'Random' else 100
    env = make_env_Rubik(step_limit=100, shuffles=shuffles, obs_type=observation_type)

    data_xy = []

    while True:
        obs = env.reset()
        episode = [cube_bin_to_str(obs)]

        solution = [move_to_action(move) for move in quarterize(
            normalize_sequence(get_raw_solution(cube_bin_to_str(obs), solver_type)))]

        for m in solution:
            obs, rew, done, info = env.step(m)
            obs_string = cube_bin_to_str(obs)
            episode.append(obs_string)

        if solver_type == 'Random':
            episode = list(reversed(episode))
            solution = [reverse_move_number(move) for move in reversed(solution)]

        for i in range(len(episode) - 1):
            dist = np.random.randint(1, 6)
            target_i = min(i + dist, len(episode) - 1)
            if brutal:
                target_i = i  # drop conditional goal
            data_xy.append(encode_policy_data(episode[i], episode[target_i], solution[i]))

        if len(data_xy) >= num_data:
            break

    print('Sample input formula', data_xy[0][0])
    print('Sample target formula', data_xy[0][1])

    random.shuffle(data_xy)

    return data_xy[:num_data]


def policy_validation(num_data, dist, model_move, final=False, brutal=False):
    env = make_env_Rubik(step_limit=100, shuffles=0 if final else 60, obs_type=observation_type)
    face_tokens, move_tokens, col_to_id, move_token_to_id = policy_encoding()

    def single_eval():
        obs = env.reset()
        target = cube_bin_to_str(obs)

        for _ in range(dist):
            obs, _, _, _ = env.step(env.action_space.sample())

        obs = cube_bin_to_str(obs)
        if obs == target:
            return 1

        for i in range(dist + 2):
            if brutal:
                policy_input = encode_policy_data(obs, obs, 0)[0]
            else:
                policy_input = encode_policy_data(obs, target, 0)[0]
            pred = model_move(policy_input)

            if len(pred) < 2:
                print('Generated invalid move:', pred)
                return 0

            move = pred[1]

            if move not in move_token_to_id:
                print('Generated invalid move:', pred)
                return 0

            move = move_token_to_id[move]
            obs, _, _, _ = env.step(move)
            obs = cube_bin_to_str(obs)
            if obs == target:
                return 1

        return 0

    return np.mean([single_eval() for _ in range(num_data)])


def dist_solver_string(state, force_solver=None):
    solver = solver_type
    if force_solver is not None:
        solver = force_solver

    try:
        solution = normalize_sequence(rubik_solver_utils.solve(state, solver))
        solution = quarterize(solution)
        return len(solution)
    except:
        return -1


def is_simple_valid_string(state):
    for c in cube_labels():
        if not state.count(c) == 9:
            return False

    return True


def is_simple_valid(state):
    state = state.reshape(54, 6)
    bin_state = np.eye(6)[np.argmax(state, axis=-1)].reshape(6, 3, 3, 6)
    s = cube_bin_to_str(bin_state)

    return is_simple_valid_string(s)


def has_fixed_colours_string(state):
    return state[4] == 'y' and state[13] == 'b' and state[22] == 'r' and \
           state[31] == 'g' and state[40] == 'o' and state[49] == 'w'


def check_valid(data):
    valid = 0
    simple_valid = 0
    fixed_valid = 0

    for i in range(len(data)):
        state = data[i]
        state = state[1:-1]

        if not len(state) == 54:
            continue

        if has_fixed_colours_string(state):
            fixed_valid += 1

        if not is_simple_valid_string(state):
            continue

        simple_valid += 1

        # print('Distances...')
        dist = dist_solver_string(state, force_solver='Kociemba')
        # print('...done:', d_in, d_out)

        if dist >= 0:
            valid += 1

    return valid / len(data), simple_valid / len(data), fixed_valid / len(data)


def eval(model, x, y, namespace):
    pred = model.predict(x)
    out = np.argmax(pred, axis=-1)
    y = np.argmax(y, axis=-1)
    res = (out == y)
    pt_acc = np.mean(res)
    st_acc = np.mean(np.min(res, axis=-1))
    # log('pointwise accuracy', pt_acc, namespace)
    # log('statewise accuracy', st_acc, namespace)

    # if observation_type == 'basic' and solver_type == 'Kociemba':
    #     check_valid(x, pred, namespace)
