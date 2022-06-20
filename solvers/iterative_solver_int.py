from queue import PriorityQueue
import random

import numpy as np

from envs import Sokoban
from envs.int.theorem_prover_env import TheoremProverEnv
from goal_builders.int.goal_builder_int import GoalBuilderINT
from solvers.core import Solver
from utils.utils_sokoban import get_field_index_from_name, HashableNumpyArray
from value_estimators.int.value_estimator_int import TrivialValueEstimatorINT
from visualization.seq_parse import logic_statement_to_seq_string


class SolverNode:
    def __init__(self, state, parent, depth, child_num, path, done, verified):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.child_num = child_num
        self.path = path
        self.done = done
        self.verified = verified
        self.children = []
        self.hash = logic_statement_to_seq_string(state['observation']['objectives'][0])

    def add_child(self, child):
        self.children.append(child)

    def set_value(self, value):
        self.value = value


class GeneralSolver(Solver):
    def __init__(self):
        self.core_env = TheoremProverEnv()


class BestFSIterativeSolverINT(GeneralSolver):
    def __init__(self,
                 goal_builder_class=None,
                 value_estimator_class=None,
                 max_tree_size=None,
                 max_tree_depth=None,
                 goal_builders_list=None,
                 iterations_list=None,
                 use_adaptive_iterations=None
                 ):
        super().__init__()
        self.max_tree_size = max_tree_size
        self.max_tree_depth = max_tree_depth
        self.value_estimator_class = value_estimator_class
        self.value_estimator = self.value_estimator_class()
        self.use_adaptive_iterations = use_adaptive_iterations

        # Either specify iterations or use adaptive scheme
        assert (iterations_list is not None
                and len(iterations_list) == len(goal_builders_list)
                and self.use_adaptive_iterations is None) \
            or (iterations_list is None
                and self.use_adaptive_iterations is not None)

        if self.use_adaptive_iterations is not None:
            assert self.use_adaptive_iterations in ['best-first', 'force-longest']

        self.policy = None

        def create_goal_builders(builder_id):  # builds models, preserving the nested list structure
            if isinstance(builder_id, list):
                return [create_goal_builders(internal_builder_id) for internal_builder_id in builder_id]
            else:
                builder = goal_builder_class(model_id=builder_id['path'], max_policy_steps=builder_id['steps'])
                if self.policy is None:
                    self.policy = builder.policy
                return builder

        self.goal_builders = create_goal_builders(goal_builders_list)
        self.iterations_list = iterations_list

        print(goal_builders_list, self.goal_builders)

    def construct_networks(self):
        self.value_estimator.construct_networks()

        def construct_goal_builders(builder):
            if isinstance(builder, list):
                for internal_builder in builder:
                    construct_goal_builders(internal_builder)
            else:
                builder.construct_networks()

        construct_goal_builders(self.goal_builders)

    def reset_counter(self):  # NOTE Disabled for goal builders.
        self.value_estimator.reset_counter()
        # for goal_builder in self.goal_builders:
        #     goal_builder.reset_counter()

    def read_counter(self):  # NOTE Disabled for goal builders.
        counters = {}
        # counters.update(self.goal_builder.read_counter())
        counters.update(self.value_estimator.read_counter())
        return counters

    def solve(self, input):
        self.reset_counter()

        assert self.value_estimator is not None, 'you must load value estimator'
        solved = False
        root = SolverNode(input, None, 0, 0, [], False, False)
        node_queues = [PriorityQueue() for _ in self.goal_builders]
        current_queue_id = 0
        # To prevent situations where two exactly same states (thus with the same value),
        # cannot be compared, we add another dimension
        # with random number which are being compared in these rare situations.
        root_value = self.value_estimator.evaluate([root.state])[0]
        root.set_value(root_value)
        for queue in node_queues:
            queue.put((-root_value, random.random(), root))
        solution = []
        tree_size = 1
        expanded_nodes = 0
        all_goals_created = 0
        tree_depth = 0
        total_path_between_goals = 0
        seen_hashed_states = {root.hash}
        real_cost = 0

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
                # print('zero_queue:', node_queues[0].queue)
                # print('top values:', top_values)
                print(f'id: {current_queue_id}, value: {top_values[current_queue_id]}')
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

            #pop node from queue to expand
            curr_val, _, current_node = node_queues[current_queue_id].get()
            print(f'val = {curr_val} | {logic_statement_to_seq_string(current_node.state["observation"]["objectives"][0])}')
            expanded_nodes += 1

            # print(logic_statement_to_seq_string(current_node.state['observation']['objectives'][0]))

            if current_node.depth < self.max_tree_depth:
                if isinstance(self.goal_builders[current_queue_id], list):
                    builders_to_expand = self.goal_builders[current_queue_id]
                else:
                    builders_to_expand = [self.goal_builders[current_queue_id]]

                goals = []
                for builder in builders_to_expand:
                    new_goals, step_real_cost = builder.build_goals(current_node.state)
                    goals += new_goals
                    real_cost += step_real_cost

                # look for solution
                for goal in goals:
                    if goal.done:
                        new_node = SolverNode(goal.subgoal_state, current_node, current_node.depth + 1, 0,
                                   goal.actions, True, goal.verified)
                        solution.append(new_node)
                        solved = True
                        tree_size += 1
                        expanded_nodes += 1
                        break

                all_goals_created += len(goals)

                created_new = 0
                subgoal_values = []
                if len(goals) > 0:
                    subgoal_values = self.value_estimator.evaluate([goal.subgoal_state for goal in goals])

                for child_num, goal_proposition in enumerate(goals):
                    current_goal_state = goal_proposition.subgoal_state
                    current_path = goal_proposition.actions
                    current_goal_state_hash = logic_statement_to_seq_string(current_goal_state['observation']['objectives'][0])
                    total_path_between_goals += len(current_path)

                    if current_goal_state_hash not in seen_hashed_states:
                        created_new += 1
                        seen_hashed_states.add(current_goal_state_hash)
                        new_node = SolverNode(current_goal_state, current_node, current_node.depth + 1, child_num,
                                              current_path, False, goal_proposition.verified)
                        current_node.add_child(new_node)
                        tree_depth = max(tree_depth, new_node.depth)
                        node_val = subgoal_values[child_num]
                        new_node.set_value(node_val)
                        for queue in node_queues:
                            queue.put((-node_val, random.random(), new_node))
                        tree_size += 1

        def save_verificator_data(builder):
            if isinstance(builder, list):
                for internal_builder in builder:
                    save_verificator_data(internal_builder)
            else:
                builder.save_verificator_data()

        save_verificator_data(self.goal_builders)

        tree_metrics = {
            'nodes': tree_size,
            'expanded_nodes': expanded_nodes,
            'unexpanded_nodes': tree_size - expanded_nodes,
            'max_depth': tree_depth,
            'avg_n_goals': all_goals_created / expanded_nodes if expanded_nodes > 0 else 0,
            'avg_dist_between_goals': total_path_between_goals / all_goals_created if all_goals_created > 0 else 0,
            'real_cost_raw': real_cost,  # not including the final trajectory
            'real_cost_final': real_cost,  # including the final trajectory
        }

        additional_info = {
            'finished_cause': finished_cause,
            'predictions':  self.read_counter(),
            'verifications': [builder.verifications for builder in self.goal_builders if not isinstance(builder, list)],
            'verificator_failed': 0,
            'calls/generator': sum([builder.calls_generator for builder in self.goal_builders if not isinstance(builder, list)]),
            'calls/verificator': sum([builder.calls_verificator for builder in self.goal_builders if not isinstance(builder, list)]),
            'calls/policy': sum([builder.calls_policy for builder in self.goal_builders if not isinstance(builder, list)]),
            'calls/value': self.value_estimator.calls_value,
            'subgoal_distances': [],
        }

        if solved:
            node = solution[0]
            while node.parent is not None:
                if node.verified:
                    base_state = node.parent.state
                    subgoal = node.state
                    result = self.policy.reach_subgoals(base_state, [subgoal], force_step_limit=5)[0]

                    if result is not None:
                        # reached
                        tree_metrics['real_cost_final'] += len(result.actions) - 1
                        additional_info['calls/policy'] += len(result.actions) * 5  # num_beams for policy
                        node.path = result.actions
                    else:
                        # not reached
                        additional_info['verificator_failed'] = 1
                        additional_info['calls/policy'] += self.policy.max_steps * 5  # num_beams for policy
                        return (None, tree_metrics, root, None, additional_info)

                additional_info['subgoal_distances'].append(len(node.path))
                solution.append(node.parent)
                node = node.parent

            trajectory_actions = []
            for inter_goal in solution:
                trajectory_actions = list(inter_goal.path) + trajectory_actions

            inter_goals = [node for node in reversed(solution)]
            return (inter_goals, tree_metrics, root, trajectory_actions, additional_info)
        else:
            return (None, tree_metrics, root, None, additional_info)

