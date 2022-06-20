import gin

from jobs.core import Job
from policies.int import vanilla_policy


class JobVanilla(Job):
    def __init__(self):
        pass

    def execute(self):
        vanilla_policy.vanilla_policy_test()
