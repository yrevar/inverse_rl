import abc

class IRLProblem(abc.ABC):

    @abc.abstractmethod
    def features(self, state):
        pass

    @abc.abstractmethod
    def sample_trajectories(self):
        pass

    @abc.abstractmethod
    def get_dynamics(self):
        pass
