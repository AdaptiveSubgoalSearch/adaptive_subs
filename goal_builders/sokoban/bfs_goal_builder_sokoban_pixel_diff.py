from collections import deque

import numpy as np

from envs import Sokoban
from goal_builders.sokoban.bfs_graph import BFSGraph
from goal_builders.sokoban.goal_builder import GoalBuilder
from goal_builders.sokoban.goal_builder_node import GoalBuilderNode
from supervised.data_creator_sokoban_pixel_diff import DataCreatorSokobanPixelDiff
from utils.general_utils import readable_num
from utils.utils_sokoban import (
    get_field_index_from_name,
    get_field_name_from_index,
    HashableNumpyArray,
)


class BFSGoalBuilderSokobanPixelDiff(GoalBuilder):
    DEFAULT_MAX_GOAL_BUILDER_TREE_DEPTH = 1000
    DEFAULT_MAX_GOAL_BUILDER_TREE_SIZE = 5000

    def __init__(
        self,
        goal_generating_network_class,
        max_goal_builder_tree_depth=None,
        max_goal_builder_tree_size=None
    ):
        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()
        self.num_boxes = self.core_env.get_num_boxes()

        self.goal_generating_network = goal_generating_network_class()
        self.max_goal_builder_tree_depth = max_goal_builder_tree_depth or self.DEFAULT_MAX_GOAL_BUILDER_TREE_DEPTH
        self.max_goal_builder_tree_size = max_goal_builder_tree_size or self.DEFAULT_MAX_GOAL_BUILDER_TREE_SIZE
        self.root = None

        self.data_creator = DataCreatorSokobanPixelDiff()
        self.elements_to_add = ['wall', 'empty', 'goal', 'box_on_goal', 'box', 'agent', 'agent_on_goal']

        self.all_nodes = []
        self.basic_edges = []
        self.extra_edges = []


    def create_root(self, input):
        root =  np.array(input, copy=True)
        return GoalBuilderNode(input, root, 1, 0, False, 0, 0, None)

    def construct_networks(self):
        self.goal_generating_network.construct_networks()

    def put_agent(self, input, x, y):
        new_state = input.copy()
        obj = np.argmax(input[x][y])
        new_state[x][y] = np.zeros(7)
        if get_field_name_from_index(obj) == 'goal':
            new_state[x][y][get_field_index_from_name('agent_on_goal')] = 1
        else:
            new_state[x][y][get_field_index_from_name('agent')] = 1
        return new_state

    def put_box(self, input, x, y):
        new_state = input.copy()
        obj = np.argmax(input[x][y])
        new_state[x][y] = np.zeros(7)
        if get_field_name_from_index(obj) == 'goal':
            new_state[x][y][get_field_index_from_name('box_on_goal')] = 1
        else:
            new_state[x][y][get_field_index_from_name('box')] = 1
        return new_state
    
    def put_board_element(self, input, x, y, element):
        new_state = input.copy()
        new_state[x][y] = np.zeros(7)
        new_state[x][y][get_field_index_from_name(element)] = 1
        return new_state

    def _generate_neighbourhood(self, initial_state, max_radius):
        return BFSGraph(self.core_env, initial_state, depth=max_radius)

    def put_element(self, x, y, state, element):
        if element in ['agent', 'agent_on_goal']:
            return self.put_agent(state, x, y)
        if element in ['box', 'box_on_goal']:
            return self.put_box(state, x, y)
        
        return self.put_board_element(state, x, y, element)

    def expand_node(self, node, pdf, internal_confidence_level, constructed_nodes):
        assert not node.done, 'node is already expanded'
        samples, probabilities = self.goal_generating_network.smart_sample(pdf, internal_confidence_level)

        for location, p in zip(samples, probabilities):
            if location[0] == self.dim_room[0]: # Model predicted end of state transformation
                node.done = True
                node.goal_state = node.condition
                node.hashed_goal = HashableNumpyArray(node.goal_state)
                continue

            element_to_add = self.elements_to_add[location[2]]

            new_state = self.put_element(location[0], location[1], node.condition, element_to_add)
            node_probability = node.p * p
            if HashableNumpyArray(new_state) in constructed_nodes.keys():
                constructed_nodes[HashableNumpyArray(new_state)].p += node_probability
                self.extra_edges.append((node.id, constructed_nodes[HashableNumpyArray(new_state)].id, readable_num(p)))
            else:
                new_node = GoalBuilderNode(
                    input_board=node.input_board,
                    condition=new_state,
                    p=node_probability,
                    elements_added=node.elements_added + 1,
                    done=False,
                    id=len(self.all_nodes),
                    level=node.level + 1,
                    parent=node
                )
                constructed_nodes[HashableNumpyArray(new_state)] = new_node
                node.children.append(new_node)
                self.all_nodes.append(new_node)
                self.basic_edges.append((node.id, new_node.id, readable_num(p)))

    def build_goals(
        self,
        input_board,
        max_radius,
        total_confidence_level,
        internal_confidence_level,
        max_goals,
        reverse_order
    ):
        goals = []
        raw_goals = self._generate_goals(internal_confidence_level, input_board)
        collected_p = 0
        accessible_goals = self._get_accessible_goals_set_paths(
            raw_goals,
            input_board,
            max_goals,
            max_radius,
            reverse_order
        )

        for goal in accessible_goals:
            goals.append(goal)
            collected_p += goal.p

            if collected_p > total_confidence_level:
                break

        return accessible_goals

    def _get_accessible_goals_set_paths(self, goals, input, max_goals, max_radius, reverse_order):
        neighbourhood_graph = self._generate_neighbourhood(input, max_radius)
        accessible_goals = [
            goal for goal in goals
            if neighbourhood_graph.is_state_visited(goal.hashed_goal)
        ]
        accessible_goals.sort(key=lambda x: x.p, reverse=reverse_order)
        accessible_goals = accessible_goals[:max_goals]

        for goal in accessible_goals:
            goal.add_path_info(neighbourhood_graph.generate_path_to_state(goal.hashed_goal))

        return accessible_goals

    def _generate_goals(self, internal_confidence_level, input):
        """
        Generates goals, but does not check if they are accessible and does not crop number of goals.
        """
        root = self.create_root(input)
        self.all_nodes.append(root)
        constructed_nodes = {}
        tree_levels = {0: [root]}
        current_level_to_expand = 0
        goals = []

        while (
            len(tree_levels[current_level_to_expand]) > 0 and
            current_level_to_expand <= self.max_goal_builder_tree_depth and
            len(constructed_nodes) <= self.max_goal_builder_tree_size
        ):
            nodes_to_expand = tree_levels[current_level_to_expand]
            input_boards = np.array([node.input_board for node in nodes_to_expand])
            conditions = np.array([node.condition for node in nodes_to_expand])
            pdfs = self.goal_generating_network.predict_pdf_batch(input_boards, conditions)
            tree_levels.setdefault(current_level_to_expand + 1, [])

            for node, pdf in zip(nodes_to_expand, pdfs):
                self.expand_node(node, pdf, internal_confidence_level, constructed_nodes)
                tree_levels[current_level_to_expand + 1] += node.children

                if node.done:
                    goals.append(node)

            current_level_to_expand += 1

        goals.sort(key=lambda x: x.p, reverse=True)

        return goals
