from typing import Union

import numpy as np

from reinforch.agents import Agent
from reinforch.core.configs import Config
from reinforch.core.logger import Log
from reinforch.models import ActorModel, CriticModel
from reinforch.utils import o2t

logging = Log(__name__)


class ActorCriticAgent(Agent):

    def __init__(self,
                 n_s: int,
                 n_a: int,
                 actor_lr: float = 0.01,
                 critic_lr: float = 0.001,
                 gamma: float = 0.99,
                 action_dim: int = None,
                 actor_config: Union[str, dict, Config] = None,
                 critic_config: Union[str, dict, Config] = None):
        super(ActorCriticAgent, self).__init__()
        self.n_s = n_s
        self.n_a = n_a
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.gamma = gamma
        self.eploration_var = 1
        self.actor_model = self.init_model(in_size=self.n_s,
                                           out_size=self.n_a,
                                           last_scale=action_dim,
                                           lr=actor_lr,
                                           config=actor_config,
                                           model_type='actor')
        self.critic_model = self.init_model(in_size=self.n_s,
                                            out_size=1,
                                            last_scale=action_dim,
                                            lr=critic_lr,
                                            gamma=gamma,
                                            config=critic_config,
                                            model_type='critic')

    def init_model(self, **kwargs):
        if 'actor' == kwargs.pop('model_type'):
            return ActorModel(**kwargs)
        elif 'critic' == kwargs.pop('model_type'):
            return CriticModel(**kwargs)

    def act(self, state):
        state = o2t(state)
        actions = self.actor_model.forward(state)
        if self.continue_action:
            action = actions.detach().numpy()[0]
            action = np.clip(np.random.normal(action, self.eploration_var), -self.action_dim, self.action_dim)
        else:
            action_prob = actions.squeeze(0).data.numpy()
            action = np.random.choice(range(self.n_a), p=action_prob)
        return action

    def step(self, state, action, reward, next_state, done, info=None, **kwargs):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param info:
        :param kwargs:
        :return:
        """
        td_error = self.critic_learn(state, reward, next_state)
        self.actor_learn(state, action, td_error)

    def critic_learn(self, state, reward, next_state):
        state = o2t(state)
        next_state = o2t(next_state)
        td_error = self.critic_model.update(state=state, reward=reward, next_state=next_state)
        return td_error

    def actor_learn(self, state, action, td_error):
        state = o2t(state)
        action = o2t([action])
        self.actor_model.update(state=state, action=action, td_error=td_error)

    def save(self, dest=None):
        self.actor_model.save(dest=dest)
        self.critic_model.save(dest=dest)

    def load(self, dest=None):
        self.actor_model.load(dest=dest)
        self.critic_model.load(dest=dest)

    def __str__(self):
        return '<ActorCriticAgent>'
