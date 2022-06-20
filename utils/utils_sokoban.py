import itertools

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
import matplotlib.pyplot as plt
import numpy as np

from metric_logging import log_image


class HashableNumpyArray:
    hash_key = np.random.normal(size=1000000)

    def __init__(self, np_array):
        assert isinstance(np_array, np.ndarray), \
            'This works only for np.array'
        assert np_array.size <= self.hash_key.size, \
            f'Expected array of size lower than {self.hash_key.size} ' \
            f'consider increasing size of hash_key.'
        self.np_array = np_array
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            flat_np = self.np_array.flatten()
            self._hash = int(np.dot(
                flat_np,
                self.hash_key[:len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)


def state_to_pic(state):
    dim_room = (state.shape[0], state.shape[1])
    env = SokobanEnvFast(dim_room, 2)
    env.restore_full_state_from_np_array_version(state)
    return env.render(mode='rgb_array').astype(int)


def save_state(state, file_name, title=None):
    pic = state_to_pic(state)
    plt.clf()
    if title is not None:
        plt.title(title)
    plt.imshow(pic)
    plt.savefig(file_name)


def many_states_to_fig(states, titles, subtitles=None):
    def draw_and_describe(plot, state, title, subtitle):
        pic = state_to_pic(state)
        plot.set_title(f'{title}', fontsize=10, pad=10)
        plot.axes.xaxis.set_ticklabels([])
        plot.axes.yaxis.set_ticklabels([])
        plot.axes.text(-5, -5, f'{subtitle}', fontsize=8)
        plot.imshow(pic)

    n_states = len(states)
    if subtitles is None:
        subtitles = ['' for _ in states]
    fig, plots = plt.subplots((n_states-1)//4 + 1, 4)
    fig.set_size_inches(4 * 2, ((n_states-1)//4 + 1) * 2.5)
    fig.set_dpi(300)
    for idx, plot in enumerate(plots.flatten()):
        if idx >= n_states:
            plot.axis('off')
            continue

        draw_and_describe(plot, states[idx], titles[idx], subtitles[idx])

    return fig


def show_state(state):
    pic = state_to_pic(state)
    plt.clf()
    plt.imshow(pic)
    plt.show()


def draw_and_describe(state, title):
    pic = state_to_pic(state)
    plt.title(f'{title}')
    plt.imshow(pic)


def draw_and_log(state, title, channel, step):
    plt.clf()
    fig = plt.figure()
    draw_and_describe(state, title)
    log_image(channel, step, fig)
    plt.close()


def get_field_name_from_index(x):
    objects = {0: 'wall', 1: 'empty', 2: 'goal', 3: 'box_on_goal', 4: 'box', 5: 'agent', 6: 'agent_on_goal'}
    return objects[x]


def get_field_index_from_name(x):
    objects_class = {'wall': 0, 'empty': 1, 'goal': 2, 'box_on_goal': 3, 'box': 4, 'agent': 5, 'agent_on_goal': 6}
    return objects_class[x]


def detect_dim_room(state):
    return (state.shape[0], state.shape[1])


def detect_num_boxes(state):
    dim_room = detect_dim_room(state)
    num_boxes = 0
    for xy in itertools.product(list(range(dim_room[0])),
                                   list(range(dim_room[1]))):
        x, y = xy
        object = get_field_name_from_index(np.argmax(state[x][y]))
        if object == 'box' or object == 'box_on_goal':
            num_boxes += 1

    return num_boxes


def action_to_agent_coordinated(action):
    """
    Returns change of agent's coordinates (-1, 0 or 1) correcponding to given action. Correspondence was taken from
    here: https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py#L166
    """
    assert action in (0, 1, 2, 3), 'Wrong value for action argument'

    translation = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    return translation[action]


def agent_coordinates_to_action(delta_x, delta_y):
    """
    Returns action corresponding to given change of agent's coordinates (-1, 0 or 1). Correspondence was taken from
    here: https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py#L166
    """
    assert delta_x in (-1, 0, 1), 'Wrong value for delta_x argument'
    assert delta_y in (-1, 0, 1), 'Wrong value for delta_y argument'

    translation = {
        (-1, 0): 0,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 3
    }
    translation_key = (delta_x, delta_y)

    assert translation_key in translation, 'Action should consists of exactly one move'

    return translation[translation_key]


def create_scheduler(drops):
    print('Creating scheduler with following drops', drops)
    return lambda epoch, lr: drops[epoch] * lr if epoch in drops else lr
