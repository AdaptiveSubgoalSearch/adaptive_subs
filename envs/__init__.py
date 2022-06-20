import gin
from envs import sokoban

def configure_env(goal_builder_class):
    return gin.external_configurable(
        goal_builder_class, module='envs'
    )

Sokoban = configure_env(sokoban.Sokoban)