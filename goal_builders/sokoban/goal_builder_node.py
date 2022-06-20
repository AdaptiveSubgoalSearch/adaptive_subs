from utils.utils_sokoban import HashableNumpyArray


class GoalBuilderNode:
    def __init__(self, input_board, condition, p, elements_added, done, id, level, parent):
        self.input_board = input_board
        self.condition = condition
        self.p = p
        self.elements_added = elements_added

        self.done = done
        self.goal_state = None
        self.hashed_goal = None

        if done:
            self.goal_state = self.condition
            self.hashed_goal = HashableNumpyArray(self.goal_state)
        self.children = []

        self.path = None
        self.id = id
        self.level = level
        self.parent = parent

        # Verificator
        self.reachable_wrt_verificator = None

    def add_path_info(self, path):
        self.path = path
