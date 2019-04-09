from typing import Union

from reinforch.core.configs import Config
from reinforch.core.memorys import Memory
from reinforch.models import DQNModel


class Agent(object):

    def __init__(self):
        pass

    def init_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def save(self, dest=None):
        raise NotImplementedError

    def load(self, dest=None):
        raise NotImplementedError


class DQNAgent(Agent):

    def __init__(self,
                 n_s: int,
                 n_a: int,
                 lr: float = 0.001,  # TODO a object
                 batch_size: int = 32,
                 memory: Memory = None,
                 action_dim: int = None,
                 double_dqn: bool = True,
                 dueling_dqn: bool = True,
                 config: Union[str, dict, Config] = None):
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
        self.model = self.init_model(input_size=n_s,
                                     output_size=n_a,
                                     last_scale=action_dim,
                                     config=config)

    def init_model(self, **kwargs):
        return DQNModel(**kwargs)

