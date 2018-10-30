class IRLProblem(object):

    def features(self, state):
        raise NotImplementedError

    def sample_trajectories(self):
        raise NotImplementedError

    def get_dynamics(self):
        raise NotImplementedError
