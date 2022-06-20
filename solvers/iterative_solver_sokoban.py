from queue import PriorityQueue
import random

import numpy as np

from envs import Sokoban
from solvers.core import Solver
from solvers.solver_sokoban import GeneralSolverSokoban
from utils.general_utils import readable_num
from utils.utils_sokoban import get_field_index_from_name, HashableNumpyArray


class SolverNode:
    def __init__(self, state, parent, p, depth, child_num, path, reachable_wrt_verificator, generator_id):
        self.state = state
        self.parent = parent
        self.p = p
        self.depth = depth
        self.child_num = child_num
        self.path = path
        self.children = []

        # Attributes needed for drawing tree:
        self.expanded = False
        self.id = None
        self.level = 0

        if self.parent is not None:
            self.trajectory_p = self.parent.trajectory_p * self.p
        else:
            self.trajectory_p = self.p

        # Verificator
        self.reachable_wrt_verificator = reachable_wrt_verificator

        self.generator_id = generator_id

    def add_child(self, child):
        self.children.append(child)

    def set_value(self, value):
        self.value = value

    # Compatibility with GoalBuilderNode
    def add_path_info(self, path):
        self.path = path

    # Compatibility with GoalBuilderNode
    @property
    def goal_state(self):
        return self.state


class BestFSIterativeSolverSokoban(GeneralSolverSokoban):
    def __init__(self,
                 goal_builder_class,
                 value_estimator_class,
                 max_steps_list,
                 max_steps_in_solution_stage,
                 total_confidence_level,
                 internal_confidence_level,
                 max_goals,
                 max_tree_size,
                 max_tree_depth,
                 goal_builders_list=None,
                 iterations_list=None,
                 use_adaptive_iterations=None,
                 num_beams=None,
                 ):
        super().__init__()
        self.max_steps_list = max_steps_list
        self.max_steps_in_solution_stage = max_steps_in_solution_stage
        self.total_confidence_level = total_confidence_level
        self.internal_confidence_level = internal_confidence_level
        self.max_goals = max_goals
        self.max_tree_size = max_tree_size
        self.max_tree_depth = max_tree_depth
        self.use_adaptive_iterations = use_adaptive_iterations
        self.use_verificator_for_solving = False
        self.num_beams = num_beams

        # Either specify iterations or use adaptive scheme
        assert (iterations_list is not None
                and len(iterations_list) == len(goal_builders_list)
                and self.use_adaptive_iterations is None) \
            or (iterations_list is None
                and self.use_adaptive_iterations is not None)

        if self.use_adaptive_iterations is not None:
            assert self.use_adaptive_iterations in ['best-first', 'force-longest']

        def create_goal_builders(builder_id):  # builds models, preserving the nested list structure
            if isinstance(builder_id, list):
                return [create_goal_builders(internal_builder_id) for internal_builder_id in builder_id]
            else:
                goal_builder = goal_builder_class(generator_id=builder_id)
                self.use_verificator_for_solving = goal_builder.use_verificator_for_solving
                return goal_builder

        self.goal_builders = create_goal_builders(goal_builders_list)
        self.iterations_list = iterations_list

        print(goal_builders_list, self.goal_builders)

        self.value_estimator = value_estimator_class()

    def construct_networks(self):
        self.value_estimator.construct_networks()

        def construct_goal_builders(builder):
            if isinstance(builder, list):
                for internal_builder in builder:
                    construct_goal_builders(internal_builder)
            else:
                builder.construct_networks()

        construct_goal_builders(self.goal_builders)

    def solve(self, input, collect_data_for_graph_tracer=False):
        assert self.value_estimator is not None, 'you must load value estimator'

        # Data needed for tree drawing (used only if collect_data_for_graph_tracer is True):
        tree_nodes = {}
        tree_edges = []
        tree_extra_edges = []
        nodes_constructed = 0

        solved = False
        root = SolverNode(input, None, 1, 0, 0, [], None, None)
        node_queues = [PriorityQueue() for _ in self.goal_builders]
        current_queue_id = 0
        # To prevent situations where two exactly same states (thus with the same value),
        # cannot be compared, we add another dimension
        # with random number which are being compared in these rare situations.
        root_value = self.value_estimator.evaluate(root.state)
        root.set_value(root_value)
        for queue in node_queues:
            queue.put((-root_value, random.random(), root))
        solution = []
        tree_size = 1
        expanded_nodes = 0
        all_goals_created = 0
        tree_depth = 0
        seen_hashed_states = {HashableNumpyArray(root.state)}
        hashed_state_to_id = {}

        total_verificator_certain_number = 0
        total_verificator_trash_number = 0
        total_verificator_to_be_verified_later_number = 0
        total_ver_calls = 0
        total_cllp_calls = 0
        total_ver_samples_in_calls = 0
        total_cllp_samples_in_calls = 0
        total_nodes_with_computations_in_graph = 1 # 1 because of root
        total_value_calls = 1 # root
        total_subgoal_gen_calls = 0

        generators_used = np.array([0 for _ in node_queues])


        current_iterations = 0

        while True:
            all_empty = True
            for queue in node_queues:
                if not queue.empty():
                    all_empty = False
                    break

            if all_empty:
                finished_cause = 'Finished cause queue is empty'
                break
            if tree_size >= self.max_tree_size:
                finished_cause = 'Finished cause tree too big'
                break
            if solved:
                finished_cause = 'Finished cause solved'
                break

            if self.use_adaptive_iterations == 'best-first':
                top_values = []

                for queue in node_queues:
                    if queue.empty():
                        value = 1e9  # infinity; never chosen as smaller is better
                    else:
                        best_elem = queue.get()
                        (value, _, _) = best_elem
                        queue.put(best_elem)
                    top_values.append(value)

                current_queue_id = np.argmin(top_values)
                print(f'id: {current_queue_id}')
            elif self.use_adaptive_iterations == 'force-longest':
                for i, queue in enumerate(node_queues):
                    if not queue.empty():
                        current_queue_id = i
                        break
                print(f'id: {current_queue_id}')
            else:
                # if the iterations limit is reached, move to the next queue
                if current_iterations >= self.iterations_list[current_queue_id]:
                    current_queue_id = (current_queue_id + 1) % len(node_queues)
                    current_iterations = 0

                # if current queue is empty, move to the next one
                while node_queues[current_queue_id].empty():
                    print(f'Queue nb {current_queue_id} is empty.')
                    current_queue_id = (current_queue_id + 1) % len(node_queues)
                    current_iterations = 0

                current_iterations += 1

            # pop node from queue to expand
            current_node = node_queues[current_queue_id].get()[-1]
            reverse_order = True  # We want goals returned by goal builder to be sorted from most to least probable.
            if current_node.depth < self.max_tree_depth:
                if isinstance(self.goal_builders[current_queue_id], list):
                    builders_to_expand = self.goal_builders[current_queue_id]
                else:
                    builders_to_expand = [self.goal_builders[current_queue_id]]

                goals = []

                for builder_id, builder in enumerate(builders_to_expand):
                    new_goals, verificator_certain_number, verificator_trash_number, verificator_to_be_verified_later_number, ver_calls, cllp_calls, ver_samples_in_calls, cllp_samples_in_calls, all_node_computations_in_graph, subgoal_gen_calls = builder.build_goals(
                        current_node.state,
                        self.max_steps_list[builder_id],
                        self.total_confidence_level,
                        self.internal_confidence_level,
                        self.max_goals,
                        reverse_order)

                    goals += new_goals

                    total_verificator_certain_number += int(verificator_certain_number)
                    total_verificator_trash_number += int(verificator_trash_number)
                    total_verificator_to_be_verified_later_number += int(verificator_to_be_verified_later_number)
                    total_ver_calls += ver_calls
                    total_cllp_calls += cllp_calls
                    total_ver_samples_in_calls += ver_samples_in_calls
                    total_cllp_samples_in_calls += cllp_samples_in_calls
                    total_nodes_with_computations_in_graph += all_node_computations_in_graph
                    total_subgoal_gen_calls += subgoal_gen_calls

                if collect_data_for_graph_tracer:
                    current_node.expanded = True

                all_goals_created += len(goals)
                expanded_nodes += 1
                created_new = 0

                for child_num, goal_proposition in enumerate(goals):
                    current_goal_state = goal_proposition.goal_state
                    current_goal_state_hashed = goal_proposition.hashed_goal
                    path = goal_proposition.path

                    if current_goal_state_hashed not in seen_hashed_states:
                        created_new += 1
                        seen_hashed_states.add(current_goal_state_hashed)
                        new_node = SolverNode(current_goal_state, current_node, goal_proposition.p,
                                              current_node.depth + 1, child_num, path,
                                              goal_proposition.reachable_wrt_verificator,
                                              current_queue_id)

                        current_node.add_child(new_node)
                        tree_depth = max(tree_depth, new_node.depth)
                        node_val = self.value_estimator.evaluate(new_node.state)
                        new_node.set_value(node_val)
                        for queue in node_queues:
                            queue.put((-node_val, random.random(), new_node))
                        tree_size += 1
                        total_value_calls += 1


                        # look for solution
                        if self.solved(current_goal_state):
                            solution.append(new_node)
                            solved = True
                            break

                    else:
                        if collect_data_for_graph_tracer:
                            extra_edge_target = hashed_state_to_id[current_goal_state_hashed]
                            tree_extra_edges.append(
                                (current_node.id, extra_edge_target, readable_num(goal_proposition.p)))
        tree_metrics = {'nodes': tree_size,
                        'expanded_nodes': expanded_nodes,
                        'unexpanded_nodes': tree_size - expanded_nodes,
                        'max_depth': tree_depth,
                        'avg_n_goals': all_goals_created / expanded_nodes if expanded_nodes > 0 else 0,
                        'verificator_failed': 0,  # To be filled during solution verification
                        'verificator_succeed': 0,  # To be filled during solution verification
                        'verificator_certain_number': total_verificator_certain_number,
                        'verificator_trash_number': total_verificator_trash_number,
                        'verificator_verified_later_number': total_verificator_to_be_verified_later_number,
                        'verificator_first_fail_goal_index': -1,  # To be filled during solution verification,
                        'ver_calls': total_ver_calls,
                        'ver_samples_in_calls': total_ver_samples_in_calls,
                        'cllp_calls': total_cllp_calls,
                        'cllp_samples_in_calls': total_cllp_samples_in_calls,
                        'total_nodes_with_computations_in_graph': total_nodes_with_computations_in_graph,
                        'value_calls': total_value_calls,
                        'subgoal_gen_calls': total_subgoal_gen_calls,
                        }
        print('Tree metrics', tree_metrics)

        additional_info = dict(
            finished_cause=finished_cause,
            tree_nodes=tree_nodes,
            tree_edges=tree_edges,
            tree_extra_edges=tree_extra_edges,
            # real_dist=np.array([np.sum(distances) if distances else 0 for distances in real_dist]),
            # real_called=real_called,
            generators_used=generators_used,
        )
        # for goal_builder in self.goal_builders:
        #     goal_builder.dump_and_clear_data_for_verificator()
        if solved:
            node = solution[0]
            while node.parent is not None:
                solution.append(node.parent)
                node = node.parent

            if self.use_verificator_for_solving:
                print('Using verificator, so have to verify final trajectories',
                      self.use_verificator_for_solving)
                # Check if solution makes sense
                initial_state = solution[-1]
                idx = 0
                for goal_state in solution[-2::-1]:  # We also skip root
                    if goal_state.reachable_wrt_verificator:
                        raw_reached, calls, samples, node_computations = \
                            self.goal_builders[goal_state.generator_id]._are_accessible_from_given_state_wrt_policy(
                            [goal_state], initial_state.state, self.max_steps_in_solution_stage, True)
                        tree_metrics['cllp_calls'] += calls
                        tree_metrics['cllp_samples_in_calls'] += samples
                        tree_metrics['total_nodes_with_computations_in_graph'] += node_computations
                        reachable = raw_reached[0]
                        if not reachable:
                            # NOTE: Abort on failure. Better try to backup.
                            tree_metrics['verificator_failed'] = 1
                            tree_metrics['verificator_first_fail_goal_index'] = idx
                            return (None, tree_metrics, root, None, additional_info)
                    initial_state = goal_state  # Update of old initial state
                    idx += 1
                tree_metrics['verificator_succeed'] = 1

            trajectory_actions = []
            for i, inter_goal in enumerate(solution):
                print('inter_goal.path', i, len(solution), f'generator_{inter_goal.generator_id}', inter_goal.path, inter_goal.reachable_wrt_verificator)
                trajectory_actions += list(inter_goal.path)
                if inter_goal.generator_id is not None:
                    generators_used[inter_goal.generator_id] += 1

            inter_goals = [node for node in reversed(solution)]
            additional_info['generators_used'] = generators_used

            return (inter_goals, tree_metrics, root, trajectory_actions, additional_info)
        else:
            return (None, tree_metrics, root, None, additional_info)
