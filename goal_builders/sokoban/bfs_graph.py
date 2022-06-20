from collections import deque

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast

from utils.utils_sokoban import HashableNumpyArray


class BFSGraph:
    class BFSNode:
        def __init__(self, state, depth, parent, parent_action):
            self.state = state
            self.depth = depth
            self.parent = parent
            self.parent_action = parent_action

    def __init__(self, env, initial_state, depth):
        assert isinstance(env, SokobanEnvFast), "Methods used for state clone and restore are sokoban specific."

        hashable_initial_state = HashableNumpyArray(initial_state)
        root = self.BFSNode(hashable_initial_state, depth=0, parent=None, parent_action=None)
        node_queue = deque([root])
        nodes = set([root])
        state2node = {hashable_initial_state: root}

        while node_queue and node_queue[0].depth < depth:
            node = node_queue.popleft()

            if node.depth > depth:
                break

            for action in range(env.action_space.n):
                env.restore_full_state_from_np_array_version(node.state.np_array)
                _, _, done, _ = env.step(action)
                neighbour_state = HashableNumpyArray(env.render(mode="one_hot"))

                # Check if state wasn't visited yet
                if neighbour_state not in state2node:
                    neighbour = self.BFSNode(neighbour_state, node.depth + 1, node, parent_action=action)
                    nodes.add(neighbour)
                    state2node[neighbour_state] = neighbour

                    if not done:
                        node_queue.append(neighbour)

        self.nodes = nodes
        self.state2node = state2node

    def generate_path_to_state(self, state):
        node = self.state2node[state]
        reversed_path = list()
        while node.depth > 0:
            reversed_path.append(node.parent_action)
            node = node.parent
        return list(reversed(reversed_path))

    def is_state_visited(self, state):
        assert isinstance(state, HashableNumpyArray)
        return state in self.state2node
