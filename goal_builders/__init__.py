import gin
from goal_builders.sokoban import (
    bfs_goal_builder_sokoban,
    bfs_goal_builder_sokoban_pixel_diff,
    policy_goal_builder_sokoban,
    policy_goal_builder_sokoban_pixel_diff,
    trivial_goal_builder_sokoban,
)

from goal_builders.int import goal_builder_int, goal_generator_int, vanilla_goal_builder_int
from goal_builders.int import mcts_vanilla_goal_builder_int, mcts_goal_builder_int
from goal_builders.shooting import goal_builder_int as shooting_goal_builder_int

from goal_builders import goal_builder_rubik, goal_generator_rubik
from goal_builders.shooting import goal_builder_shooting_rubik


def configure_goal_builder(goal_builder_class):
    return gin.external_configurable(
        goal_builder_class, module='goal_builders'
    )


BFSGoalBuilderSokoban = configure_goal_builder(bfs_goal_builder_sokoban.BFSGoalBuilderSokoban)
BFSGoalBuilderSokobanPixelDiff = configure_goal_builder(bfs_goal_builder_sokoban_pixel_diff.BFSGoalBuilderSokobanPixelDiff)
PolicyGoalBuilderSokobanPixelDiff = configure_goal_builder(policy_goal_builder_sokoban_pixel_diff.PolicyGoalBuilderSokobanPixelDiff)
PolicyGoalBuilderSokoban = configure_goal_builder(policy_goal_builder_sokoban.PolicyGoalBuilderSokoban)
TrivialGoalBuilderSokoban = configure_goal_builder(trivial_goal_builder_sokoban.TrivialGoalBuilderSokoban)


GoalBuilderINT = configure_goal_builder(goal_builder_int.GoalBuilderINT)
GoalGeneratorINT = configure_goal_builder(goal_generator_int.GoalGeneratorINT)
SamplingGoalGeneratorINT = configure_goal_builder(goal_generator_int.SamplingGoalGeneratorINT)
VanillaGoalBuilderINT = configure_goal_builder(vanilla_goal_builder_int.VanillaGoalBuilderINT)

MCTSVanillaGoalBuilderInt = configure_goal_builder(
    mcts_vanilla_goal_builder_int.MCTSVanillaGoalBuilderInt
)
MCTSGoalBuilderInt = configure_goal_builder(
    mcts_goal_builder_int.MCTSGoalBuilderInt
)

GoalBuilderForShootingINT = configure_goal_builder(
    shooting_goal_builder_int.GoalBuilderForShootingINT
)

GoalBuilderRubik = configure_goal_builder(goal_builder_rubik.GoalBuilderRubik)
GoalGeneratorRubik = configure_goal_builder(goal_generator_rubik.GoalGeneratorRubik)
SamplingGoalGeneratorRubik = configure_goal_builder(goal_generator_rubik.SamplingGoalGeneratorRubik)

GoalBuilderForShootingRubik = configure_goal_builder(goal_builder_shooting_rubik.GoalBuilderForShootingRubik)
