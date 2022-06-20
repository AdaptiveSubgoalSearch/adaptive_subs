import time

from jobs.core import Job
from policies import SokobanPolicyBaseline
from supervised import DataCreatorPolicyBaselineSokoban
from utils.general_utils import readable_num


class JobSokobanTrainPolicyBaseline(Job):
    def __init__(self, dataset, dump_folder, epochs, epochs_checkpoints):
        self.policy = SokobanPolicyBaseline()
        self.dataset = dataset
        self.dump_folder = dump_folder
        self.epochs = epochs
        self.epochs_checkpoints = epochs_checkpoints

        self.data_creator = DataCreatorPolicyBaselineSokoban()

    def execute(self):
        total_time_start = time.time()

        self.policy.construct_networks()
        self.data_creator.load(self.dataset)
        x_train, y_train, x_validation, y_validation = self.data_creator.create_train_and_validation_sets()

        self.policy.fit_and_dump(
            x_train,
            y_train,
            x_validation,
            y_validation,
            self.epochs,
            self.dump_folder,
            self.epochs_checkpoints
        )

        return readable_num(time.time() - total_time_start)