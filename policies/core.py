class GeneralPolicy:
    def act(self, observation):
        raise NotImplementedError

class TargetPolicy:
    def act(self, observation, target):
        raise NotImplementedError