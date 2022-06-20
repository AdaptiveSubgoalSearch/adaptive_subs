import gin
from goal_generating_networks import conditional_goal_predictor_sokoban
from goal_generating_networks import goal_predictor_pixel_diff

def configure_goal_generating_network(goal_builder_class):
    return gin.external_configurable(
        goal_builder_class, module='goal_generating_networks'
    )

ConditionalGoalPredictorSokoban = configure_goal_generating_network(conditional_goal_predictor_sokoban.ConditionalGoalPredictorSokoban)
GoalPredictorPixelDiff = configure_goal_generating_network(goal_predictor_pixel_diff.GoalPredictorPixelDiff)