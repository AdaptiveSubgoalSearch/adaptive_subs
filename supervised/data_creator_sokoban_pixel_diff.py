import itertools
import pickle
import random
import joblib

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
from joblib import Parallel, delayed
from metric_logging import log_scalar
import numpy as np
from os.path import join, isdir
from os import listdir
from tqdm import tqdm

from utils.utils_sokoban import get_field_index_from_name, \
    get_field_name_from_index, detect_dim_room, detect_num_boxes


class TrajectoryHandler:
    """
    A helper class for batching the reachability checks.

    Produces a stream of state-subgoal pairs to be checked after
    consecutive calls to the next_request method.

    For every state on the trajectory it determines the best subgoal.
    Currently, the one just before the first non-achievable on the trajectory.

    Includes optimistic-check optimization: for every state the search
    for subgoal starts from the subgoal for the previous state if
    achievable. This may slightly change the actual results, but
    seems to be beneficial in such case.
    """
    def __init__(self, x, key=None):
        self.x = x  # the trajectory to process
        self.y = np.arange(len(x))  # subgoal ids; initially match the states
        self.key = key  # key of the trajectory in dataset, used for logging only
        self.prev_request = (0, 0)  # previous request yielded
        self.is_optimistic = False  # flags whether opimistic check was asked
        self.completed = False  # flags whether the computations are completed

    def next_request(self, prev_reached, progress=None):
        if self.completed:
            return None

        (state_idx, subgoal_idx) = self.prev_request

        if prev_reached:
            # check the next subgoal
            self.is_optimistic = False
            self.y[state_idx] = subgoal_idx
            subgoal_idx += 1

        subgoal_distance = subgoal_idx - state_idx

        if not prev_reached or subgoal_idx == len(self.x):
            # subgoal was not reached or the end of the trajectory was reached
            if self.is_optimistic:
                # it was an optimistic check; proceed the standard way
                self.is_optimistic = False
                subgoal_idx = state_idx + 1
            else:
                # non-achievable subgoal found; proceed to the next state
                state_idx += 1
                subgoal_idx = max(subgoal_idx - 1, state_idx + 1)
                self.is_optimistic = True

        if progress is not None:
            # update progress bar
            progress[0] += state_idx - self.prev_request[0]

        if state_idx == len(self.x) - 1:
            self.completed = True

        if self.completed:
            return None
        else:
            self.prev_request = (state_idx, subgoal_idx)
            return self.x[state_idx], self.x[subgoal_idx], min(subgoal_idx - state_idx + 2, 30)  # no more than 30 steps

    def get_subgoals(self, verbose=False):
        assert self.completed
        if verbose:
            print(f'Subgoals on trajectory {self.key}', list(self.y))
            for i in range(len(self.x)):
                print(f'subgoal dist: {self.y[i] - i} of {len(self.x) - 1 - i}')
        return self.x, [self.x[i] for i in self.y]


class DataCreatorSokobanPixelDiff:
    def __init__(self,
                 validation_split=None,
                 keep_trajectories=None,
                 keep_samples=None,
                 n_parallel_workers=None,
                 batch_size=None,
                 use_adaptive_subgoals=False,
                 conditional_policy_network_class=None,
                 num_checks=1,
                 verification='greedy'):

        self.validation_split = validation_split
        self._keep_trajectories = keep_trajectories
        self._keep_samples = keep_samples
        self._n_parallel_workers = n_parallel_workers
        self._batch_size = batch_size
        self.data = {}
        self.training_keys = None
        self.validation_keys = None
        self.dim_room = None
        self.num_boxes = None

        self.use_adaptive_subgoals = use_adaptive_subgoals
        self.conditional_policy = None
        self.num_checks = num_checks
        self.verification = verification
        if use_adaptive_subgoals:
            self.conditional_policy = conditional_policy_network_class()
            self.conditional_policy.construct_networks()

    def load(self, dataset_path=None):
        if isdir(dataset_path):
            files = listdir(dataset_path)
            for file in files:
                print(f'Loading data from file {file}.')
                part_dict = joblib.load(join(dataset_path, file))
                self.data.update(part_dict)
        else:
            with open(dataset_path, 'rb') as handle:
                self.data = pickle.load(handle)

        all_keys_shuffled = list(self.data.keys()).copy()
        random.shuffle(all_keys_shuffled)
        val_split_num = int(len(all_keys_shuffled) * self.validation_split) + 1
        self.validation_keys = all_keys_shuffled[:val_split_num]
        self.training_keys = all_keys_shuffled[val_split_num:]
        assert len(self.training_keys) > 0

        self.dim_room = detect_dim_room(self.data[self.training_keys[0]][0])
        self.num_boxes = detect_num_boxes(self.data[self.training_keys[0]][0])

        self.log_env = SokobanEnvFast(self.dim_room, self.num_boxes)
        self.log_env.reset()

    def get_reachable_subgoals(self, x, distance_distribution):
        # The reachability check for the given policy
        # Increase the distance to the subgoal until reachable
        # Batch queries for a single trajectory
        # NOTE currently not used, can be treated as an additional verification
        y = np.arange(len(x))
        fixed = np.array([False] * len(x))
        distance = 0

        while not np.all(fixed):
            distance += 1
            y += 1 - fixed  # increase for not fixed states
            fixed = np.logical_or(fixed, y == len(x))  # fix for all that reached the end
            active_idx = np.where(np.array(fixed) == False)[0]

            states = [x[i] for i in active_idx]
            subgoals = [x[y[i]] for i in active_idx]

            can_reach = self.conditional_policy.can_reach_batched(states=states, subgoals=subgoals,
                                                                  steps=[distance + 2] * len(states))

            for i, reachable in zip(active_idx, can_reach):
                if not reachable:
                    fixed[i] = True

        y = y - 1  # shift from the first non-reachable to the last reachable

        distances = y - np.arange(len(x))
        max_distances = len(x) - 1 - np.arange(len(x))
        for d, max_d in zip(distances, max_distances):
            print(f'subgoal dist: {d} of {max_d}')
            distance_distribution[d] = distance_distribution.get(d, 0) + 1

        return y

    def create_xy(self, steps, keys):
        print(f'running create_xy with {steps} steps and {keys} keys.')
        assert self.training_keys is not None and \
               self.validation_keys is not None, 'You must load data first.'

        assert keys == 'train' or keys == 'validate', \
            f'keys accepts values: train or validate, got {keys}.'

        keys_to_dataset = {'train': self.training_keys,
                           'validate': self.validation_keys
                           }

        global_x = []
        global_y = []

        distance_distribution = {}
        handlers = []
        active_handlers = []
        responses = []
        num_datapoints = 0

        # choose trajectories, store as handlers
        for key in keys_to_dataset[keys]:
            if random.random() < self._keep_trajectories:
                x = list(self.data[key]).copy()
                num_datapoints += len(x) - 1
                active_handlers.append(len(handlers))
                handlers.append(TrajectoryHandler(x, key))
                responses.append(True)

        if self.use_adaptive_subgoals:
            # compute the subgoals in all the trajectories
            num_step = 0
            progress = [0]

            while True:
                num_step += 1

                states = []
                subgoals = []
                steps = []
                new_active_handlers = []

                for i, handler_idx in enumerate(active_handlers):
                    request = handlers[handler_idx].next_request(responses[i], progress=progress)
                    if request is None:
                        continue

                    new_active_handlers.append(handler_idx)
                    state, subgoal, step = request
                    for _ in range(self.num_checks):
                        states.append(state)
                        subgoals.append(subgoal)
                        steps.append(step)

                if not new_active_handlers:
                    print(f'Adaptive subgoals completed in {num_step} steps.')
                    break

                responses = self.conditional_policy.can_reach_batched(states=states, subgoals=subgoals, steps=steps,
                                                                      batch_size=1000,
                                                                      sampling=self.verification)
                responses = np.min(np.array(responses).reshape((-1, self.num_checks)), axis=-1)  # accept only if all runs reached the target

                active_handlers = new_active_handlers
                print(f'Step {num_step} completed. {len(active_handlers)} trajectories remain. Progress: {progress[0] / num_datapoints * 100 :.0f}%')

        for idx, handler in enumerate(handlers):
            if self.use_adaptive_subgoals:
                x, y = handler.get_subgoals(verbose=(idx < 100))
            else:
                x = list(handler.x).copy()
                y = list(handler.x).copy()

                last = y[-1]
                y = y[steps:]
                y = y + [last] * min(steps, len(x))
            # Now, ys are the subgoals for the corresponding xs

            all_elems = len(x)
            elems_to_keep = int(len(x) * self._keep_samples)
            indices_to_keep = random.sample(range(all_elems), elems_to_keep)

            x = [x[i] for i in indices_to_keep]
            y = [y[i] for i in indices_to_keep]

            global_x.extend(x)
            global_y.extend(y)

        print(f'Dataset size has {len(global_x)} elements.')
        print('Distance distribution:', distance_distribution)
        global_x = np.array(global_x)
        global_y = np.array(global_y)
        return global_x, global_y

    def create_x(self):
        global_x = []
        for num in tqdm(range(len(self.data))):
            x = list(self.data[num]).copy()
            global_x = global_x + x

        return global_x

    def remove_one_box(self, state_np):
        box = None
        new_state = np.zeros(state_np.shape)

        box_removed = False
        for xy in itertools.product(list(range(self.dim_room[0])),
                                    list(range(self.dim_room[1]))):
            x, y = xy
            object = get_field_name_from_index(np.argmax(state_np[x][y]))
            if box_removed == False:
                if object == 'box':
                    box = xy
                    new_state[x][y][get_field_index_from_name('empty')] = 1
                    box_removed = True
                elif object == 'box_on_goal':
                    box = xy
                    new_state[x][y][get_field_index_from_name('goal')] = 1
                    box_removed = True
                else:
                    new_state[x][y][get_field_index_from_name(object)] = 1
            else:
                new_state[x][y][get_field_index_from_name(object)] = 1
        return box, new_state

    def remove_agent(self, state_np):
        agent = None
        new_state = np.zeros(state_np.shape)

        for xy in itertools.product(list(range(self.dim_room[0])),
                                    list(range(self.dim_room[1]))):
            x, y = xy
            object = get_field_name_from_index(np.argmax(state_np[x][y]))
            if object == 'agent':
                agent = xy
                new_state[x][y][get_field_index_from_name('empty')] = 1
            elif object == 'agent_on_goal':
                agent = xy
                new_state[x][y][get_field_index_from_name('goal')] = 1
            else:
                new_state[x][y][get_field_index_from_name(object)] = 1

        return agent, new_state

    def split_goal(self, goal):

        goal_partials = []
        objects_coordinates = []
        state = goal
        for _ in range(self.num_boxes):
            box_xy, state = self.remove_one_box(state)
            goal_partials.append(state)
            objects_coordinates.append(self.flatten_xy(box_xy))

        agent_xy, state = self.remove_agent(state)
        goal_partials.append(state)
        objects_coordinates.append(self.flatten_xy(agent_xy))

        return reversed(goal_partials), reversed(objects_coordinates)

    def clear(self, state):
        if self.dim_room is None:
            self.dim_room = detect_dim_room(state)
            self.num_boxes = detect_num_boxes(state)
        _, state = self.remove_one_box(state)
        _, state = self.remove_one_box(state)
        _, state = self.remove_agent(state)
        return state

    def flatten_xy(self, xy):
        output = np.zeros(shape=self.dim_room)
        x, y = xy
        output[x][y] = 1
        size = self.dim_room[0] * self.dim_room[1]
        return output.reshape(1, size)[0]

    def create_xy_split(self, steps, keys):

        data_x, data_y = self.create_xy(steps, keys)
        global_x_input = []
        global_x_condition = []
        global_y = []

        batch_num = 0
        processed = 0
        while processed < len(data_x):
            print(f'processed {processed} = '
                  f'{int(1000 * processed / len(data_x)) / 10} %')

            log_scalar(f'data/processing_{keys}', batch_num,
                       int(1000 * processed / len(data_x)) / 10)

            results = Parallel(
                n_jobs=self._n_parallel_workers, verbose=1)(
                delayed(split_board_objects)(data_y[num], data_x[num])
                for num in range(processed, processed +
                                 min(self._batch_size, len(data_x) - processed))
            )
            processed += self._batch_size
            batch_num += 1

            for result in results:

                input_board, partial_goals, targets = result

                for state, target in zip(partial_goals, targets):
                    global_x_input.append(input_board)
                    global_x_condition.append(state)
                    global_y.append(target)

        global_x_input = np.array(global_x_input)
        global_x_condition = np.array(global_x_condition)
        global_y = np.array(global_y)

        return global_x_input, global_x_condition, global_y


def remove_object(object, state, xy):
    x, y = xy
    state[x][y][get_field_index_from_name(object)] = 0
    if object == 'box_on_goal' or object == 'agent_on_goal':
        state[x][y][get_field_index_from_name('goal')] = 1
    elif object == 'box' or object == 'agent':
        state[x][y][get_field_index_from_name('empty')] = 1

    return state


def split_board_objects(state, input_board):
    # dim_room = state.shape[:2]
    # agent = None
    # boxes = []
    targets = []

    # semi_states = []

    semi_state = input_board.copy()
    semi_states = [semi_state.copy()]
    for xy in itertools.product(list(range(state.shape[0])),
                                list(range(state.shape[1]))):

        x, y = xy
        # object_on_board = get_field_name_from_index(np.argmax(state[x][y]))
        if (state[x, y] != input_board[x, y]).any():
            semi_state[x, y] = state[x, y]
            semi_states.append(semi_state.copy())
            target = np.zeros(shape=semi_state.shape)
            target[x, y] = state[x, y]
            target = target.flatten()
            target = np.concatenate([target, [0]])
            targets.append(target)
            a=1

    # Last target: predict that no more modifications are needed.
    targets.append(
        np.concatenate([
            np.zeros(shape=semi_state.shape).flatten(),
            [1]
        ])
    )
    return input_board, semi_states, targets



    # random.shuffle(boxes)
    # semi_state = np.array(state, copy=True)
    # removed_boxes = []
    #
    # while boxes:
    #     current_box = boxes.pop()
    #     removed_boxes.append(current_box[0])
    #     targets.append(tuple(removed_boxes))
    #     semi_state = np.array(semi_state, copy=True)
    #     if current_box[1] == False:
    #         semi_state = remove_object('box', semi_state, current_box[0])
    #     else:
    #         semi_state = remove_object('box_on_goal', semi_state,
    #                                    current_box[0])
    #     semi_states.append(semi_state)
    #
    # targets.append(tuple([agent[0]]))
    # if agent[1] == False:
    #     semi_state = np.array(semi_state, copy=True)
    #     semi_state = remove_object('agent', semi_state, agent[0])
    # else:
    #     semi_state = np.array(semi_state, copy=True)
    #     semi_state = remove_object('agent_on_goal', semi_state, agent[0])
    # semi_states.append(semi_state)
    #
    # targets_np = [target_to_np(target, dim_room) for target in targets]
    #
    # return input_board, list(reversed(semi_states)), list(reversed(targets_np))


def clear_board(state):
    new_state = np.zeros(state.shape)

    for xy in itertools.product(list(range(state.shape[0])), list(range(state.shape[1]))):
        x, y = xy
        field_name = get_field_name_from_index(np.argmax(state[x][y]))

        if field_name in ['box', 'agent']:
            new_state[x][y][get_field_index_from_name('empty')] = 1
        elif field_name in ['box_on_goal', 'agent_on_goal']:
            new_state[x][y][get_field_index_from_name('goal')] = 1
        else:
            new_state[x][y][get_field_index_from_name(field_name)] = 1

    return new_state


def flatten_positions(list_xy, shape):
    def flat_one_position(xy):
        output = np.zeros(shape=shape)
        x, y = xy
        output[x][y] = 1
        size = shape[0] * shape[1]
        return output.reshape(1, size)[0]

    return [flat_one_position(xy) for xy in list_xy]


def target_to_np(target, shape):
    output = np.zeros(shape=shape)
    for point in target:
        x, y = point
        output[x][y] = 1
    output = output / len(target)
    size = shape[0] * shape[1]
    return output.reshape(1, size)[0]
