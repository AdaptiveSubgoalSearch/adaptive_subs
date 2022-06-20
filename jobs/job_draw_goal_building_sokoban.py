import time

from envs import Sokoban
from goal_builders import BFSGoalBuilderSokoban
from graph_tracer import GraphTracerSokoban
from jobs.core import Job
from supervised import DataCreatorSokoban
from utils.general_utils import readable_num
from utils.utils_sokoban import draw_and_log, many_states_to_fig
from metric_logging import log_image, log_scalar

class JobDrawGoalBuildingSokoban(Job):
    def __init__(self,
                 goal_builder_class,
                 total_confidence_level,
                 internal_confidence_level,
                 max_goals,
                 max_radius,
                 ):

        self.goal_builder = goal_builder_class()
        self.total_confidence_level = total_confidence_level
        self.internal_confidence_level = internal_confidence_level
        self.max_goals = max_goals
        self.max_radius = max_radius

        self.data_creator = DataCreatorSokoban()
        self.core_env = Sokoban()

        self.graph_tarcer = GraphTracerSokoban()

    def execute(self):

        self.goal_builder.construct_networks()
        input = self.core_env.reset()
        total_time_start = time.time()
        self.goal_builder.build_goals(input,
                                      self.max_radius,
                                      self.total_confidence_level,
                                      self.internal_confidence_level,
                                      self.max_goals,
                                      True)
        finished_time = readable_num(time.time() - total_time_start)
        edges = self.goal_builder.basic_edges + self.goal_builder.extra_edges
        self.graph_tarcer.draw_goal_generation(self.goal_builder.all_nodes, edges)

        print(f'goal building took = {finished_time}, constructed = {len(self.goal_builder.all_nodes)} nodes')

        return finished_time