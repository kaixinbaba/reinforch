from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optimize

from reinforch.core.networks import Network


class Model(object):
    def __init__(self, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def from_config(config):
        pass

    def save(self, dest=None):
        raise NotImplementedError

    def load(self, dest=None):
        raise NotImplementedError

    def close(self):
        pass


class DQNModel(Model):

    def __init__(self,
                 in_size=None,
                 out_size=None,
                 last_scale=None,
                 lr=0.001,
                 gamma=0.99,
                 c_step=1000,
                 soft_update=False,
                 tau=0.01,
                 config=None):
        super(DQNModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.last_scale = last_scale
        self.lr = lr
        self.gamma = gamma
        self.c_step = c_step
        self.learn_count = 0
        self.soft_update = soft_update
        self.tau = tau
        self.eval_network = Network.from_config(config=config)
        self.target_net = deepcopy(self.eval_network)
        self.loss = F.mse_loss
        self.optim = optimize.Adam(self.eval_network.parameters(), lr=lr)

    def update(self, b_s=None, b_a=None, b_r=None, b_s_=None, b_done=None):
        eval_q = torch.gather(self.eval_network(b_s), 1, b_a)
        next_max_from_eval_index = self.eval_network(b_s_).max(1)[1].unsqueeze(1)
        # fix target net
        next_actions = self.target_net(b_s_).detach()
        next_max = next_actions.gather(1, next_max_from_eval_index)

        target_q = b_r + self.gamma * next_max * (1 - b_done)
        loss = self.loss(eval_q, target_q)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._update_target_net()

    def _soft_update(self):
        for target_param, local_param in zip(
                self.target_net.parameters(), self.eval_network.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update(self):
        self.target_net.load_state_dict(self.eval_network.state_dict())

    def _update_target_net(self):
        if self.soft_update:
            self._soft_update()
        else:
            if self.learn_count % self.c_step == 0:
                self._hard_update()
