"""Neural network trainers."""

import gin

from alpacka.trainers import dummy
from alpacka.trainers import supervised
from alpacka.trainers import td
from alpacka.trainers import union
from alpacka.trainers.base import *


# Configure trainers in this module to ensure they're accessible via the
# alpacka.trainers.* namespace.
def configure_trainer(trainer_class):
    return gin.external_configurable(
        trainer_class, module='alpacka.trainers'
    )


DummyTrainer = configure_trainer(dummy.DummyTrainer)  # pylint: disable=invalid-name
SupervisedTrainer = configure_trainer(supervised.SupervisedTrainer)  # pylint: disable=invalid-name
TDTrainer = configure_trainer(td.TDTrainer)  # pylint: disable=invalid-name
UnionTrainer = configure_trainer(union.UnionTrainer)  # pylint: disable=invalid-name
