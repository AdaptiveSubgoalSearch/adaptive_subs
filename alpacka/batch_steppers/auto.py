"""Environment steppers."""

import os

from alpacka.batch_steppers import local
from alpacka.batch_steppers import ray


if 'LOCAL_RUN' in os.environ:
    class AutoBatchStepper(local.LocalBatchStepper):
        pass
else:
    class AutoBatchStepper(ray.RayBatchStepper):
        pass
