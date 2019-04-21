from typing import Union

import numpy as np

from reinforch.agents import Agent
from reinforch.core.configs import Config
from reinforch.core.logger import Log
from reinforch.models import PolicyGradientModel
from reinforch.utils import o2t, LongTensor

logging = Log(__name__)


class PolicyGradientAgent(Agent):

    def __init__(self,
                 n_s: int,
                 n_a: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 memory=None,
                 action_dim: int = None,
                 config: Union[str, dict, Config] = None):
        super(PolicyGradientAgent, self).__init__()
        self.n_s = n_s
        self.n_a = n_a
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        self.eploration_var = 1
        self.model = self.init_model(in_size=self.n_s,
                                     out_size=self.n_a,
                                     last_scale=action_dim,
                                     lr=lr,
                                     gamma=gamma,
                                     config=config)

    def init_model(self, **kwargs):
        return PolicyGradientModel(**kwargs)

    def act(self, state):
        state = o2t(state)
        actions = self.model.forward(state)
        if self.continue_action:
            action = actions.detach().numpy()[0]
            action = np.clip(np.random.normal(action, self.eploration_var), -self.action_dim, self.action_dim)
        else:
            action_prob = actions.squeeze(0).data.numpy()
            action = np.random.choice(range(self.n_a), p=action_prob)
        return action

    def step(self, state, action, reward, next_state, done, info=None, **kwargs):
        """
        因为PG是回合更新的，所以只存储数据，不更新参数

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param info:
        :param kwargs:
        :return:
        """

        self.memory.store(state=state,
                          action=action,
                          reward=reward)

    def _normalize_rewards(self, rewards):
        # FIXME 写的太丑
        l = list(map(lambda t: t[0] * (self.gamma ** t[1]), zip(rewards, range(0, len(rewards)))))
        rewards = np.reshape(np.array(
            list(map(lambda x: x[0], l))),
            (-1, 1))
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        return rewards

    def learn(self):
        states, actions, rewards = self.memory.sample().get('mini_batch')

        # 转成tensor对象
        states = o2t(states)
        if self.continue_action:
            actions = o2t(actions)
        else:
            actions = o2t(actions, target_type=LongTensor)
        rewards = o2t(self._normalize_rewards(rewards))
        self.model.update(states=states, actions=actions, rewards=rewards)

    def after_done(self):
        self.learn()

    def save(self, dest=None):
        self.model.save(dest=dest)

    def load(self, dest=None):
        self.model.load(dest=dest)

    def __str__(self):
        return '<PolicyGradientAgent>'
