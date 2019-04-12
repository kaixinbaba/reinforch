from typing import Union

import numpy as np

from reinforch.core.configs import Config
from reinforch.core.logger import Log
from reinforch.core.memorys import Memory
from reinforch.models import DQNModel
from reinforch.utils import o2t

logging = Log(__name__)


class Agent(object):

    def __init__(self):
        pass

    def init_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def step(self,
             state,
             action,
             reward,
             next_state,
             done,
             info=None,
             **kwargs):
        raise NotImplementedError

    def _learn(self):
        pass

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
                 lr: float = 0.001,  # TODO an object
                 gamma: float = 0.99,
                 c_step: int = 1000,
                 soft_update: bool = False,
                 tau: float = 0.01,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.99,
                 epsilon_min: float = 0.0,
                 batch_size: int = 32,
                 memory: Memory = None,
                 learn_threshold: int = None,
                 action_dim: int = None,
                 double_dqn: bool = True,
                 dueling_dqn: bool = True,
                 config: Union[str, dict, Config] = None):
        super(DQNAgent, self).__init__()
        self.n_s = n_s
        self.n_a = n_a
        self.lr = lr
        self.gamma = gamma
        self.c_step = c_step
        self.soft_update = soft_update
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_need_decay = True
        self.batch_size = batch_size
        self.memory = memory
        self.learn_threshold = learn_threshold
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.model = self.init_model(input_size=n_s,
                                     output_size=n_a,
                                     last_scale=action_dim,
                                     lr=lr,
                                     gamma=gamma,
                                     c_step=c_step,
                                     soft_update=soft_update,
                                     tau=tau,
                                     config=config)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # random choose an action
            return np.random.randint(self.n_a)
        else:
            # greedy choose max action
            # cast to pytorch tensor
            state = o2t(state)
            action = self.model.forward(state)
            # cast an action into an environment needed.
            return action

    def step(self,
             state,
             action,
             reward,
             next_state,
             done,
             info=None,
             **kwargs):
        self.memory.store(state=state,
                          action=action,
                          reward=reward,
                          next_state=next_state,
                          done=done,
                          **kwargs)
        can_learn = (self.learn_threshold is not None
                     and len(self.memory) > self.learn_threshold) \
                    or len(self.memory) > self.batch_size
        if not can_learn:
            return
        self._learn()
        self._update_epsilon()

    def _update_epsilon(self):
        if not self.epsilon_need_decay:
            return
        epsilon = self.epsilon * self.epsilon_decay
        if epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
            self.epsilon_need_decay = False
        else:
            self.epsilon = epsilon

    def _learn(self):
        b_state, b_action, b_reward, b_next_state, b_done = self.memory.sample(self.batch_size)

        self.model.update(b_s=b_state,
                          b_a=b_action,
                          b_r=b_reward,
                          b_s_=b_next_state,
                          b_done=b_done)

    def init_model(self, **kwargs):
        return DQNModel(**kwargs)

    def save(self, dest=None):
        self.model.save(dest=dest)

    def load(self, dest=None):
        self.model.load(dest=dest)

    def close(self):
        self.model.close()
