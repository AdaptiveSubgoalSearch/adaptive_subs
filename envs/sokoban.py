from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast


class Sokoban(SokobanEnvFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dim_room(self):
        return self.dim_room

    def get_num_boxes(self):
        return self.num_boxes
