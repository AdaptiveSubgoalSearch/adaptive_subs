import time

from jobs.core import Job
from policies import SokobanConditionalPolicy
from supervised import DataCreatorConditionalPolicySokoban
from utils.general_utils import readable_num


class JobSokobanTrainConditionalPolicy(Job):
    def __init__(self, dataset, dump_folder, epochs, epochs_checkpoints):
        self.policy = SokobanConditionalPolicy()
        self.dataset = dataset
        self.dump_folder = dump_folder
        self.epochs = epochs
        self.epochs_checkpoints = epochs_checkpoints

        self.data_creator = DataCreatorConditionalPolicySokoban()

    def execute(self):
        total_time_start = time.time()

        self.policy.construct_networks()
        self.data_creator.load(self.dataset)

        self.policy.fit_and_dump(
            self.data_creator,
            self.epochs,
            self.dump_folder,
            self.epochs_checkpoints
        )

        return readable_num(time.time() - total_time_start)