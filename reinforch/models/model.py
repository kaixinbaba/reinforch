from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optimize

from reinforch.core.networks import Network


class Model(object):
    """
    真正的算法模型对象，Agent大部分算法都委托给它来处理.

    Model对象内部可以组合多个神经网络进行训练学习

    """

    def __init__(self, **kwargs):
        pass

    def forward(self, x):
        """
        神经网络正向传播流程.

        从输入到输出的过程
        :param x: 整个模型的输入
        :return: 整个模型的输出
        """
        raise NotImplementedError

    def update(self, **kwargs):
        """
        神经网络反向传播流程, 由子类实现.

        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def from_config(config):
        pass

    def save(self, dest: str = None):
        """
        保存模型参数.

        :param dest: 目标保存文件路径
        """

        raise NotImplementedError

    def load(self, dest: str = None):
        """
        读取模型参数.

        :param dest: 目标文件路径
        """

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
                 double_dqn=True,
                 dueling_dqn=True,
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
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.eval_network = Network.from_config(config=config)
        self.target_net = deepcopy(self.eval_network)
        self.loss = F.mse_loss
        self.optim = optimize.Adam(self.eval_network.parameters(), lr=lr)

    def forward(self, x):
        return self.eval_network(x)

    def update(self, b_s=None, b_a=None, b_r=None, b_s_=None, b_done=None):
        eval_q = torch.gather(self.eval_network(b_s), 1, b_a)
        if self.double_dqn:
            next_max_action_index = self._double_choose_max_action(b_s_)
        else:
            next_max_action_index = self._normal_choose_max_action(b_s_)

        # fix target net
        next_actions = self.target_net(b_s_).detach()
        next_max = next_actions.gather(1, next_max_action_index)

        target_q = b_r + self.gamma * next_max * (1 - b_done)
        abs_errors = torch.sum(torch.abs(target_q - eval_q), dim=1).detach().numpy()
        loss = self.loss(eval_q, target_q)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._update_target_net()
        return abs_errors

    def _normal_choose_max_action(self, b_s_):
        return self.target_net(b_s_).max(1)[1].unsqueeze(1)

    def _double_choose_max_action(self, b_s_):
        return self.eval_network(b_s_).max(1)[1].unsqueeze(1)

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

    def save(self, dest: str = 'dqn_model.pkl'):
        """
        保存整个网络的模型及参数

        :param dest:
        :return:
        """

        torch.save(self.eval_network.state_dict(), dest)

    def load(self, dest: str = 'dqn_model.pkl'):
        """
        读取文件加载网络的模型及参数

        :param dest:
        :return:
        """

        self.eval_network.load_state_dict(torch.load(dest))
        self.target_net = deepcopy(self.eval_network)


class PolicyGradientModel(Model):

    def __init__(self,
                 in_size=None,
                 out_size=None,
                 last_scale=None,
                 lr=0.001,
                 gamma=0.99,
                 config=None):
        super(PolicyGradientModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.last_scale = last_scale
        self.lr = lr
        self.gamma = gamma
        self.learn_count = 0
        self.network = Network.from_config(config=config)
        self.optim = optimize.Adam(self.network.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)

    def update(self, states=None, actions=None, rewards=None):
        if self.last_scale is None:
            actions_prop = self.network(states)
            actions = torch.gather(actions_prop, 1, actions)
        else:
            actions = self.network(states)
            # actions = actions
        loss = torch.mean(-torch.log(actions) * rewards)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def save(self, dest: str = None):
        """
        保存整个网络的模型及参数

        :param dest:
        :return:
        """

        torch.save(self.network.state_dict(), dest)

    def load(self, dest: str = None):
        """
        读取文件加载网络的模型及参数

        :param dest:
        :return:
        """

        self.network.load_state_dict(torch.load(dest))


class ActorModel(Model):

    def __init__(self,
                 in_size=None,
                 out_size=None,
                 last_scale=None,
                 lr=0.01,
                 gamma=0.99,
                 config=None):
        super(ActorModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.last_scale = last_scale
        self.lr = lr
        self.gamma = gamma
        self.learn_count = 0
        self.network = Network.from_config(config=config)
        self.optim = optimize.Adam(self.network.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)

    def update(self, state=None, action=None, td_error=None):
        action_prop = self.network(state)
        action_value = torch.gather(action_prop, 1, action)
        actor_loss = torch.mean(-torch.log(action_value) * td_error)

        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()

    @staticmethod
    def __actor_save_path(dest):
        return 'actor_{}'.format(dest)

    def save(self, dest: str = None):
        """
        保存整个网络的模型及参数

        :param dest:
        :return:
        """

        torch.save(self.network.state_dict(), self.__actor_save_path(dest))

    def load(self, dest: str = None):
        """
        读取文件加载网络的模型及参数

        :param dest:
        :return:
        """

        self.network.load_state_dict(torch.load(self.__actor_save_path(dest)))


class CriticModel(Model):

    def __init__(self,
                 in_size=None,
                 out_size=None,
                 last_scale=None,
                 lr=0.001,
                 gamma=0.99,
                 config=None):
        super(CriticModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.last_scale = last_scale
        self.lr = lr
        self.gamma = gamma
        self.learn_count = 0
        self.network = Network.from_config(config=config)
        self.optim = optimize.Adam(self.network.parameters(), lr=lr)
        self.loss = F.mse_loss

    def forward(self, x):
        return self.network(x)

    def update(self, state=None, reward=None, next_state=None):
        V = self.network(state)
        V_ = self.network(next_state)
        V_target = reward + self.gamma * V_
        td_error = V_target - V
        critic_loss = self.loss(V_target, V)
        self.optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optim.step()
        return td_error

    @staticmethod
    def __critic_save_path(dest):
        return 'critic_{}'.format(dest)

    def save(self, dest: str = None):
        """
        保存整个网络的模型及参数

        :param dest:
        :return:
        """

        torch.save(self.network.state_dict(), self.__critic_save_path(dest))

    def load(self, dest: str = None):
        """
        读取文件加载网络的模型及参数

        :param dest:
        :return:
        """

        self.network.load_state_dict(torch.load(self.__critic_save_path(dest)))
