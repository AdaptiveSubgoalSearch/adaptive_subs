import numpy as np

from gym_rubik.envs.cube import Actions, Cube


class CubeletSet:
    def __init__(self, colours_list, assign_table, is_even=None):
        self.count = len(colours_list)
        self.colours = colours_list
        self.dim = len(colours_list[0])
        self.assign_table = assign_table
        self.is_even = [False] * self.count if is_even is None else is_even

        self.ids = None
        self.position_table = None

        self.make_ids()
        self.make_position_table()

    def make_ids(self):
        self.ids = dict()

        for i in range(self.count):
            self.ids[self.colours[i]] = i

    def make_position_table(self):
        self.position_table = [sorted([tuple(place) for place in np.transpose(np.where(self.assign_table == i))]) for i in range(self.count)]

    def encode(self, observation):
        res = np.zeros((self.count, 24), dtype=np.float)

        for i in range(self.count):
            position = self.position_table[i]
            colours = [observation[place] for place in position]
            colours_sorted = tuple(sorted(colours))
            id = self.ids[colours_sorted]
            res[id, self.dim * i + np.argmin(colours)] = 1.

        return res

    def decode(self, observation, result):
        for i in range(self.count):
            idx = np.where(observation[i] == 1)[0][0]
            place = idx // self.dim
            rotation = idx % self.dim

            colours = self.colours[i]
            colours_rotated = [0] * self.dim
            step_direction = -1 if self.is_even[i] ^ self.is_even[place] else 1

            for j in range(self.dim):
                colours_rotated[(rotation + j * step_direction) % self.dim] = colours[j]

            for k, pos in enumerate(self.position_table[place]):
                result[pos] = colours_rotated[k]


class CubeConverter:
    def __init__(self, debug=False):
        self.debug = debug

        x = -1
        self.corners = CubeletSet(
            colours_list=[(0, 2, 5), (0, 3, 5), (0, 2, 4), (0, 3, 4), (1, 2, 5), (1, 3, 5), (1, 2, 4), (1, 3, 4)],
            is_even=[False, True, True, False, True, False, False, True],
            assign_table=np.array(
                [[[0, x, 1],
                  [x, x, x],
                  [2, x, 3]],

                 [[5, x, 4],
                  [x, x, x],
                  [7, x, 6]],

                 [[4, x, 0],
                  [x, x, x],
                  [6, x, 2]],

                 [[7, x, 3],
                  [x, x, x],
                  [5, x, 1]],

                 [[6, x, 2],
                  [x, x, x],
                  [7, x, 3]],

                 [[5, x, 1],
                  [x, x, x],
                  [4, x, 0]]]),
        )

        self.edges = CubeletSet(
            colours_list=[(0, 5), (0, 2), (0, 3), (0, 4), (2, 5), (3, 5), (2, 4), (3, 4), (1, 5), (1, 2), (1, 3), (1, 4)],
            assign_table=np.array(
                [[[x, 0, x],
                  [1, x, 2],
                  [x, 3, x]],

                 [[x, 8, x],
                  [10, x, 9],
                  [x, 11, x]],

                 [[x, 4, x],
                  [9, x, 1],
                  [x, 6, x]],

                 [[x, 7, x],
                  [10, x, 2],
                  [x, 5, x]],

                 [[x, 6, x],
                  [11, x, 3],
                  [x, 7, x]],

                 [[x, 5, x],
                  [8, x, 0],
                  [x, 4, x]]]),
        )

    def convert_basic_to_reduced(self, basic_observation, force_no_debug=False):
        result = np.concatenate([self.corners.encode(basic_observation), self.edges.encode(basic_observation)], axis=0)

        if self.debug and not force_no_debug:
            print("converter debug")
            assert (np.array_equal(basic_observation, self.convert_reduced_to_basic(result, force_no_debug=True)))

        return result

    def convert_reduced_to_basic(self, reduced_observation, force_no_debug=False):
        result = np.zeros((6, 3, 3), dtype=np.float32)

        self.corners.decode(reduced_observation[:self.corners.count, :], result)
        self.edges.decode(reduced_observation[self.corners.count:, :], result)

        for i in range(6):
            result[i, 1, 1] = i

        if self.debug and not force_no_debug:
            print("converter debug")
            assert (np.array_equal(reduced_observation, self.convert_basic_to_reduced(result, force_no_debug=True)))

        return result


def test_converter():
    ACTION_LOOKUP = {
        0: Actions.U,
        1: Actions.U_1,
        2: Actions.D,
        3: Actions.D_1,
        4: Actions.F,
        5: Actions.F_1,
        6: Actions.B,
        7: Actions.B_1,
        8: Actions.R,
        9: Actions.R_1,
        10: Actions.L,
        11: Actions.L_1
    }

    c = CubeConverter(debug=True)

    for i in range(100):
        cube = Cube(3, whiteplastic=False)
        for _ in range(100):
            cube.move_by_action(ACTION_LOOKUP[np.random.randint(0, 12)])
        # print(c.convert_basic_to_reduced(cube.get_state()))

        if not np.array_equal(cube.get_state(), c.convert_reduced_to_basic(c.convert_basic_to_reduced(cube.get_state()))):
            print("TESTING FAILED")
            print(c.convert_basic_to_reduced(cube.get_state()))
            print(cube.get_state())
            print(c.convert_reduced_to_basic(c.convert_basic_to_reduced(cube.get_state())))
            assert (False)

    print("TESTING OK")
