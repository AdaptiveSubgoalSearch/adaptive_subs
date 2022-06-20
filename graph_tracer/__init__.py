import gin

from graph_tracer import graph_tracer_sokoban


def configure_graph_tracer(goal_builder_class):
    return gin.external_configurable(
        goal_builder_class, module='goal_builders'
    )

GraphTracerSokoban = configure_graph_tracer(graph_tracer_sokoban.GraphTracerSokoban)