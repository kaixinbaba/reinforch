class Agent(object):
    """
    所有Agent的父类， 具体算法子类Agent必须继承该类.

    """

    def __init__(self):
        pass

    def init_model(self, **kwargs):
        """
        初始化算法模型对象.

        :param kwargs:
        :return: 神经网络模型对象
        """
        raise NotImplementedError

    def act(self, state):
        """
        根据状态选择一个动作, 由子类实现.

        :param state: 环境给出的状态
        :return: 动作
        """

        raise NotImplementedError

    def step(self,
             state,
             action,
             reward,
             next_state,
             done,
             info=None,
             **kwargs):
        """
        进行一次迭代(存储记忆，更新参数等), 由子类实现.

        :param state: 状态
        :param action: 动作
        :param reward: 奖励
        :param next_state: 下一状态
        :param done: 是否结束
        :param info: 备注信息
        :param kwargs:
        """

        raise NotImplementedError

    def after_done(self):
        """
        在每一个episode结束后的行为，通常是回合更新agent 学习的时候

        :return:
        """
        pass

    def learn(self):
        """
        智能体学习更新参数, 由子类实现

        :return:
        """

        raise NotImplementedError

    def _after_update(self, *args, **kwargs):
        """
        模型更新后，钩子方法

        :param args:
        :param kwargs:
        :return:
        """

        pass

    def close(self):
        """
        关闭智能体，释放资源等.

        """

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

    def __str__(self):
        raise NotImplementedError
