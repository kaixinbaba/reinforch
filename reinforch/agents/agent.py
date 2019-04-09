class Agent(object):

    def __init__(self):
        pass

    def init_model(self):
        pass

    def act(self, state):
        pass

    def step(self):
        pass

    def close(self):
        pass

    def save(self, dest=None):
        pass

    def load(self, dest=None):
        pass


class DQNAgent(Agent):

    def __init__(self,
                 n_s: int,
                 n_a: int,
                 lr=0.001,  # TODO a object
                 batch_size=32,
                 memory=None,
                 action_dim: int=None,
                 double_dqn:bool=True,
                 dueling_dqn:bool=True):
        super(DQNAgent, self).__init__()
        self.n_s = n_s
        self.n_a = n_a
        self.lr = lr
        self.batch_size = batch_size
        self.memory = memory
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
