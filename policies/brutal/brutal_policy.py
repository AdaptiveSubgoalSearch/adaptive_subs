from policies.brutal.axiom_selector import search_for_subgoal_bfs


class BrutalPolicy:
    def __init__(self, max_distance):
        self.max_distance = max_distance

    def reach_subgoal(self, state, subgoal_str):
        ground_truth = state["observation"]["ground_truth"]
        return search_for_subgoal_bfs(state, subgoal_str, ground_truth, self.max_distance, False)

