from abc import ABC


class GoalBuilder(ABC):
    def build_goals(
        self,
        input_board,
        max_radius,
        total_confidence_level,
        internal_confidence_level,
        max_goals,
        reverse_order
    ):
        raise NotImplementedError()
    
    def construct_networks(self):
        pass
