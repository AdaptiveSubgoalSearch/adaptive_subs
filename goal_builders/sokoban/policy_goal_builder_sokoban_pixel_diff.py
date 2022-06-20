from collections import deque
from ctypes import ArgumentError
from PIL.Image import init
import joblib

import numpy as np
import tensorflow.keras as tfk
import tensorflow as tf
import random

from tensorflow.python.ops.math_ops import to_bfloat16
from envs import Sokoban
from goal_builders.sokoban.goal_builder import GoalBuilder
from goal_builders.sokoban.goal_builder_node import GoalBuilderNode
from supervised.data_creator_sokoban_pixel_diff import DataCreatorSokobanPixelDiff
from utils.general_utils import readable_num
from utils.utils_sokoban import (
    get_field_index_from_name,
    get_field_name_from_index,
    HashableNumpyArray,
)


class PolicyGoalBuilderSokobanPixelDiff(GoalBuilder):
    DEFAULT_MAX_GOAL_BUILDER_TREE_DEPTH = 1000
    DEFAULT_MAX_GOAL_BUILDER_TREE_SIZE = 5000
    DEFAULT_NUM_BEAMS = 8

    def __init__(
            self,
            goal_generating_network_class,
            conditional_policy_network_class,
            max_goal_builder_tree_depth=None,
            max_goal_builder_tree_size=None,
            max_batch_size=32,
            gather_data_for_verificator=False,
            max_data_points_per_part=100_000,
            use_verificator_for_solving=False,
            verificator_ckpt_path=None,
            verificator_batch_size=32,
            ver_trash_th=0.0,
            ver_certainty_th=0.9,
            num_beams=None,
            use_beam_search=False,
            generator_id=None,
    ):
        self.max_bs = max_batch_size
        self.core_envs = [Sokoban() for _ in range(max_batch_size)]
        self.dim_room = self.core_envs[0].get_dim_room()
        self.num_boxes = self.core_envs[0].get_num_boxes()

        if generator_id is None:
            self.goal_generating_network = goal_generating_network_class()
        else:
            self.goal_generating_network = goal_generating_network_class(model_id=generator_id)
        self.conditional_policy_network = conditional_policy_network_class()
        self.max_goal_builder_tree_depth = max_goal_builder_tree_depth or self.DEFAULT_MAX_GOAL_BUILDER_TREE_DEPTH
        self.max_goal_builder_tree_size = max_goal_builder_tree_size or self.DEFAULT_MAX_GOAL_BUILDER_TREE_SIZE
        self.num_beams = num_beams or self.DEFAULT_NUM_BEAMS
        self.use_beam_search = use_beam_search
        self.root = None

        self.data_creator = DataCreatorSokobanPixelDiff()
        self.elements_to_add = ['wall', 'empty', 'goal', 'box_on_goal', 'box', 'agent', 'agent_on_goal']

        self.all_nodes = []
        self.basic_edges = []
        self.extra_edges = []

        # Params for verificator training
        self.goal_builder_id = random.randint(1, 30_000)
        self.gather_data_for_verificator = gather_data_for_verificator
        self.verificator_data = []
        self.gathered_data_points_number = 0
        self.verificator_data_epochs = 0
        self.max_data_points_per_part = max_data_points_per_part

        # Params for verificator solving
        self.use_verificator_for_solving = use_verificator_for_solving
        self.verificator_ckpt_path = verificator_ckpt_path
        self.verificator_batch_size = verificator_batch_size
        self.ver_trash_th = ver_trash_th
        self.ver_certainty_th = ver_certainty_th

        if (self.use_verificator_for_solving and not self.verificator_ckpt_path):
            raise ArgumentError("Can't use verificator if no checkpoint")

        if self.use_verificator_for_solving and (self.ver_trash_th < 0 and self.ver_certainty_th > 1):
            #raise ArgumentError("None of tresholds is in [0, 1]")
            self.use_verificator_for_solving = False

        if self.use_verificator_for_solving:
            self.verificator = tfk.models.load_model(self.verificator_ckpt_path)

    def create_root(self, input):
        root = np.array(input, copy=True)
        return GoalBuilderNode(input, root, 1, 0, False, 0, 0, None)

    def construct_networks(self):
        self.goal_generating_network.construct_networks()
        self.conditional_policy_network.construct_networks()

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

    def _check_if_goals_are_reachable_wrt_verificator(self, goals, initial_state):
        if not self.verificator:
            raise Exception('Verificator is not loaded. Cant be used')
        left = len(goals)
        node_computations = len(goals)

        data_for_verificator = [np.concatenate([initial_state, subgoal.goal_state], axis=-1) for subgoal in goals]

        all_raw_ver_predictions = tf.zeros((0,)) # empty tensor

        # Verificator processing
        calls = 0
        while left > 0:
            current_bs = min(left, self.verificator_batch_size)
            current_batch = np.array(data_for_verificator[len(goals)-left:len(goals)-left+current_bs])
            calls += 1
            batch_raw_ver_predictions = tf.squeeze(self.verificator(current_batch), axis=[1]) # (bs, )
            all_raw_ver_predictions = tf.concat([all_raw_ver_predictions, batch_raw_ver_predictions], axis=0)
            left -= current_bs

        # Converting to the List[bool]
        trash_mask = all_raw_ver_predictions <= self.ver_trash_th
        certainty_mask = all_raw_ver_predictions >= self.ver_certainty_th
        tobecheck_mask = np.logical_and(all_raw_ver_predictions > self.ver_trash_th, all_raw_ver_predictions < self.ver_certainty_th)

        verificator_certain_number_in_batch = tf.math.count_nonzero(certainty_mask)
        verificator_trash_number_in_batch = tf.math.count_nonzero(trash_mask)
        verificator_to_be_verified_later = tf.math.count_nonzero(tobecheck_mask)

        return certainty_mask, tobecheck_mask, verificator_certain_number_in_batch, verificator_trash_number_in_batch, verificator_to_be_verified_later, calls, node_computations

    def _are_accessible_from_given_state_wrt_policy_and_verificator(self, goals, initial_state, max_radius):
        node_computations = 0

        # Gather mask from verificator
        certainty_mask, tobecheck_mask, verificator_certain_number_in_batch, verificator_trash_number_in_batch, verificator_to_be_verified_later_number_in_batch, ver_calls, node_computations_from_ver = self._check_if_goals_are_reachable_wrt_verificator(goals, initial_state)

        node_computations += node_computations_from_ver

        tbc_idxs = np.where(tobecheck_mask)[0]
        cer_idxs = np.where(certainty_mask)[0]
        not_cer_idxs = np.where(np.logical_not(certainty_mask))[0]

        # Perform CLLP checks
        goals_cllp = [goals[idx] for idx in tbc_idxs]

        ver_samples_in_calls = len(goals)

        raw_reached, cllp_calls, cllp_samples_in_calls, node_computations_from_cllp = self._are_accessible_from_given_state_wrt_policy(goals_cllp, initial_state, max_radius, True)

        node_computations += node_computations_from_cllp

        reached_cllp = np.array(raw_reached, dtype=np.bool)
        reached_idxs = np.where(reached_cllp)[0]

        # Construct final mask
        cllp_decision_mask = np.full(tobecheck_mask.shape, False, dtype=bool)
        cllp_reached_mask = np.full((len(tbc_idxs), ), False, dtype=bool)
        cllp_reached_mask[reached_idxs] = True
        cllp_decision_mask[tbc_idxs] = cllp_reached_mask # cllp_decision_mask[tbc_idxs] has length of goals_cllp

        final_mask = np.logical_or(certainty_mask, cllp_decision_mask)

        # Update information about goals
        for g in [goals[idx] for idx in cer_idxs]:
            g.reachable_wrt_verificator = True
        for g in [goals[idx] for idx in not_cer_idxs]:
            g.reachable_wrt_verificator = False

        return final_mask.tolist(), verificator_certain_number_in_batch, verificator_trash_number_in_batch, verificator_to_be_verified_later_number_in_batch, ver_calls, cllp_calls, ver_samples_in_calls, cllp_samples_in_calls, node_computations


    def _are_accessible_from_given_state_wrt_policy(self, goals, initial_state, max_radius, used_after_verificator):
        bs = len(goals)

        node_computations = 0

        current_states = [initial_state.copy() for i in range(bs)]
        envs = self.core_envs[:bs]
        paths = [[] for _ in range(bs)]

        # Indexes of active states
        assert (len(current_states) == len(goals))
        reached = [np.array_equal(state, subgoal.goal_state) for state, subgoal in zip(current_states, goals)]
        active_idxs = np.where(np.array(reached) == False)[0]
        reached_without_moves_idxs = np.where(np.array(reached) == True)[0]

        # For each reached state at the beginning, we have to add empty path
        for i in reached_without_moves_idxs:
            goals[i].add_path_info([])

        # For each unreached state, we have to initialize it
        for i in active_idxs:
            envs[i].restore_full_state_from_np_array_version(initial_state)

        # Processing
        calls = 0
        cllp_samples_in_calls = 0
        for i in range(max_radius):
            active_idxs = np.where(np.array(reached) == False)[0]
            if len(active_idxs) == 0:
                assert (len(np.where(np.array(reached) == True)[0]) == bs)
                break

            batch = np.array([np.concatenate([current_states[i], goals[i].goal_state], axis=-1) for i in active_idxs])
            calls += 1
            actions = np.argmax(self.conditional_policy_network.predict_action_batch(batch), axis=-1)
            cllp_samples_in_calls += len(actions)

            if not used_after_verificator:
                node_computations += len(actions)
            elif used_after_verificator and i != 0:
                # We don't want to add first batch to node computations because those states where already called with verificator
                node_computations += len(actions)

            for i, action in zip(active_idxs, actions):
                paths[i].append(action)  # Update of path
                current_states[i], _, _, _ = envs[i].step(action)  # Update of current state
                reached[i] = np.array_equal(current_states[i], goals[i].goal_state)
                if reached[i]:
                    goals[i].add_path_info(paths[i])

        return reached, calls, cllp_samples_in_calls, node_computations

    def put_element(self, x, y, state, element):
        if element in ['agent', 'agent_on_goal']:
            return self.put_agent(state, x, y)
        if element in ['box', 'box_on_goal']:
            return self.put_box(state, x, y)

        return self.put_board_element(state, x, y, element)

    def get_locations_and_probabilities(self, node, pdf, node_id):
        locations, probabilities = self.goal_generating_network.sample(pdf)
        locations_with_probabilities = [(locations[i], probabilities[i], probabilities[i] * node.p, node_id) for i in
                                        range(len(locations))]
        return locations_with_probabilities

    def choose_candidates(self, probs_for_all_nodes):
        probs_for_all_nodes.sort(key=lambda x: x[2], reverse=True)
        probs_for_all_nodes = probs_for_all_nodes[:min(len(probs_for_all_nodes), self.num_beams)]
        return probs_for_all_nodes

    def expand_node_for_one_sample(self, node, location, p, constructed_nodes):
        if location[0] == self.dim_room[0]:  # Model predicted end of state transformation
            node.done = True
            node.goal_state = node.condition
            node.hashed_goal = HashableNumpyArray(node.goal_state)
            return True

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

        return False

    def expand_node_for_all_samples(self, node, pdf, internal_confidence_level, constructed_nodes):
        assert not node.done, 'node is already expanded'
        samples, probabilities = self.goal_generating_network.sample_with_internal_confidence_level(pdf,
                                                                                                    internal_confidence_level)
        prob_for_end_of_transform = 0.0
        for location, p in zip(samples, probabilities):
            if self.expand_node_for_one_sample(node, location, p, constructed_nodes):
                prob_for_end_of_transform = p

        return prob_for_end_of_transform

    def build_goals(
            self,
            input_board,
            max_radius,
            total_confidence_level,
            internal_confidence_level,
            max_goals,
            reverse_order
    ):
        all_node_computations_in_graph = 0

        goals = []
        if self.use_beam_search:
            raw_goals = self._generate_goals_with_beam_search(input_board)
        else:
            raw_goals = self._generate_goals(internal_confidence_level, input_board)

        collected_p = 0

        subgoal_gen_calls = self.num_beams

        raw_goals = raw_goals[:max_goals]

        accessible_goals, verificator_certain_number, verificator_trash_number, verificator_to_be_verified_later_number, ver_calls, cllp_calls, ver_samples_in_calls, cllp_samples_in_calls, node_computations_from_ver_and_cllp = self._get_accessible_goals_set_paths(
            raw_goals,
            input_board,
            max_goals,
            max_radius,
            reverse_order
        )

        all_node_computations_in_graph += node_computations_from_ver_and_cllp

        for goal in accessible_goals:
            goals.append(goal)
            collected_p += goal.p

            if collected_p > total_confidence_level:
                break
        return accessible_goals, verificator_certain_number, verificator_trash_number, verificator_to_be_verified_later_number, ver_calls, cllp_calls, ver_samples_in_calls, cllp_samples_in_calls, all_node_computations_in_graph, subgoal_gen_calls

    def _get_accessible_goals_set_paths(self, goals, input, max_goals, max_radius, reverse_order):
        if not self.use_verificator_for_solving:
            goals.sort(key=lambda x: x.p, reverse=reverse_order)
            goals = goals[:max_goals]

        left = len(goals)
        accessible_goals = []

        node_computations_from_ver_and_cllp = 0

        verificator_certain_number = 0
        verificator_trash_number = 0
        verificator_to_be_verified_later_number = 0

        ver_calls = 0
        cllp_calls = 0
        ver_samples_in_calls = 0
        cllp_samples_in_calls = 0
        while left > 0:
            bs = min(self.max_bs, left)  # Current batch size
            goals_in_batch = goals[len(goals) - left:len(goals) - left + bs]

            if not self.use_verificator_for_solving:
                accessible_mask, cllp_calls_in_batch, cllp_samples_in_calls_in_batch, node_computations_from_cllp = self._are_accessible_from_given_state_wrt_policy(
                    goals_in_batch, input, max_radius, False)

                cllp_calls += cllp_calls_in_batch
                cllp_samples_in_calls += cllp_samples_in_calls_in_batch
                node_computations_from_ver_and_cllp += node_computations_from_cllp
            else:
                accessible_mask, verificator_certain_number_in_batch, verificator_trash_number_in_batch, verificator_to_be_verified_later_number_in_batch, ver_calls_in_batch, cllp_calls_in_batch, ver_samples_in_calls_in_batch, cllp_samples_in_calls_in_batch, nodes_computations = self._are_accessible_from_given_state_wrt_policy_and_verificator(
                    goals_in_batch, input, max_radius)

                verificator_certain_number += verificator_certain_number_in_batch
                verificator_trash_number += verificator_trash_number_in_batch
                verificator_to_be_verified_later_number += verificator_to_be_verified_later_number_in_batch
                ver_calls += ver_calls_in_batch
                cllp_calls += cllp_calls_in_batch
                ver_samples_in_calls += ver_samples_in_calls_in_batch
                cllp_samples_in_calls += cllp_samples_in_calls_in_batch
                node_computations_from_ver_and_cllp += nodes_computations

            accessible_goals_patch = [goals_in_batch[j] for j in range(bs) if accessible_mask[j]]
            accessible_goals += accessible_goals_patch

            left -= bs

        accessible_goals.sort(key=lambda x: x.p, reverse=reverse_order)

        return accessible_goals, verificator_certain_number, verificator_trash_number, verificator_to_be_verified_later_number, ver_calls, cllp_calls, ver_samples_in_calls, cllp_samples_in_calls, node_computations_from_ver_and_cllp

    def dump_and_clear_data_for_verificator(self):
        if not self.gather_data_for_verificator:
            return

        joblib.dump(self.verificator_data, f'builder{self.goal_builder_id}_{self.verificator_data_epochs}', compress=5)
        self.verificator_data_epochs += 1
        self.verificator_data = []

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
                prob_for_end_of_transform = self.expand_node_for_all_samples(node, pdf, internal_confidence_level, constructed_nodes)
                tree_levels[current_level_to_expand + 1] += node.children

                if node.done:
                    node.p = node.p * prob_for_end_of_transform
                    goals.append(node)

            current_level_to_expand += 1

        goals.sort(key=lambda x: x.p, reverse=True)

        return goals

    def _generate_goals_with_beam_search(self, input):
        """
        Generates goals, but does not check if they are accessible and does not crop number of goals.
        """
        root = self.create_root(input)
        self.all_nodes.append(root)
        constructed_nodes = {}
        tree_levels = {0: [root]}
        current_level_to_expand = 0
        goals = []
        worst_score = 1e-9


        while (
                len(goals) <= self.num_beams and
                len(tree_levels[current_level_to_expand]) > 0 and
                current_level_to_expand <= self.max_goal_builder_tree_depth
        ):
            nodes_to_expand = tree_levels[current_level_to_expand]
            input_boards = np.array([node.input_board for node in nodes_to_expand])
            conditions = np.array([node.condition for node in nodes_to_expand])
            pdfs = self.goal_generating_network.predict_pdf_batch(input_boards, conditions)

            tree_levels.setdefault(current_level_to_expand + 1, [])

            probs_for_all_nodes = []
            for node_id, (node, pdf) in enumerate(zip(nodes_to_expand, pdfs)):
                probs_for_all_nodes += self.get_locations_and_probabilities(node, pdf, node_id)

            candidates = self.choose_candidates(probs_for_all_nodes)

            if len(goals) >= self.num_beams and all([c[2] <= worst_score for c in candidates]):
                break

            expanded_nodes_ids = []
            end_of_transformation_info = {}

            for (location, prob, _, node_id) in candidates:
                end_of_transform_occured = self.expand_node_for_one_sample(nodes_to_expand[node_id], location, prob, constructed_nodes)

                if end_of_transform_occured:
                    end_of_transformation_info[node_id] = prob

                if node_id not in expanded_nodes_ids:
                    expanded_nodes_ids.append(node_id)

            for node_id, node in enumerate(nodes_to_expand):
                if node_id in expanded_nodes_ids:
                    tree_levels[current_level_to_expand + 1] += nodes_to_expand[node_id].children

                if node.done and (len(goals) < self.num_beams or node.p > worst_score):
                    node.p = node.p * end_of_transformation_info[node_id]
                    goals.append(node)

                    if len(goals) > self.num_beams:
                        goals_sorted_by_probs = sorted([(goal.p, idx) for idx, goal in enumerate(goals)])
                        del goals[goals_sorted_by_probs[0][1]]
                        worst_score = goals_sorted_by_probs[1][0]
                    else:
                        worst_score = min(node.p, worst_score)

            current_level_to_expand += 1

        goals.sort(key=lambda x: x.p, reverse=True)
        return goals
