import time

from goal_generating_networks import ConditionalGoalPredictorSokoban, \
    GoalPredictorPixelDiff
from jobs.core import Job
from supervised import DataCreatorSokoban
from utils.general_utils import readable_num


class JobTrainSokoban(Job):
    def __init__(self,
                 dataset,
                 dump_folder,
                 steps_into_future,
                 epochs,
                 epochs_checkpoints
                 ):

        self.goal_generating_network = ConditionalGoalPredictorSokoban()
        self.dataset = dataset
        self.dump_folder = dump_folder
        self.steps_into_future = steps_into_future
        self.epochs = epochs
        self.epochs_checkpoints = epochs_checkpoints

        self.data_creator = DataCreatorSokoban()

    def execute(self):
        total_time_start = time.time()

        self.goal_generating_network.construct_networks()
        self.data_creator.load(self.dataset)
        x_input, x_condition, ys = \
            self.data_creator.create_xy_split(self.steps_into_future, 'train')
        vx_input, vx_condition, vys = \
            self.data_creator.create_xy_split(self.steps_into_future, 'validate')
        self.goal_generating_network.fit_and_dump([x_input, x_condition],
                                                  ys, ([vx_input, vx_condition], vys),
                                                  self.epochs, self.dump_folder, self.epochs_checkpoints)

        return readable_num(time.time() - total_time_start)