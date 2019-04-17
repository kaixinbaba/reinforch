from reinforch.agents import Agent


class DQNAgent(Agent):
    """
    Deep-Q-Net Agent算法及其变种实现.

    DQN暂时只支持离散动作的环境, 面对连续动作空间，建议使用其他算法
    相关算法
    TargetNet
    ReplayBuffer
    PrioritizeReplayBuffer
    DoubleDQN
    DuelingDQN
    Rainbow
    """

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
        """

        :param n_s: 状态空间数量
        :param n_a: 动作空间数量
        :param lr:  学习率
        :param gamma: 折扣率
        :param c_step: 每隔多少步同步target_net参数，仅当soft_update:False生效
        :param soft_update: 是否软更新, True:每次迭代均会同步target_net一部分参数，由tau控制
                                      False: 每隔c_step更新target_net参数
        :param tau: 每次同步多少target_net参数，仅当soft_update:True生效
        :param epsilon: epsilon_greedy算法选取动作核心参数，小于该参数为随机选择动作，大于该参数为贪婪策略，选取期望回报最大的动作
        :param epsilon_decay: epsilon衰减率
        :param epsilon_min: 最小epsilon，衰减到该值就不再衰减
        :param batch_size: 控制每次sample样本数量，建议选取2的幂次方
        :param memory: DQN算法用于存储训练样本的记忆库
        :param learn_threshold: 当memory存储数量大于该值时，模型开始学习
        :param action_dim: 动作选取区间值（该值取区间上限）仅当动作为连续有效
        :param double_dqn: 是否采用double dqn算法,减少Q-learning过估计问题
        :param dueling_dqn: 是否采用dueling dqn算法,使用优势函数替代Q值函数，以平衡动作和状态
        :param config: 是否采取配置初始化 TODO Agent对象也可以使用配置来初始化
        """

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
        self.model = self.init_model(in_size=self.n_s,
                                     out_size=self.n_a,
                                     last_scale=action_dim,
                                     lr=lr,
                                     gamma=gamma,
                                     c_step=c_step,
                                     soft_update=soft_update,
                                     tau=tau,
                                     double_dqn=double_dqn,
                                     dueling_dqn=dueling_dqn,
                                     config=config)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # random choose an action
            return np.random.randint(self.n_a)
        else:
            # cast to pytorch tensor
            state = o2t(state)
            action = self.model.forward(state)
            # select greedy action
            return int(action.argmax())

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
        """
        更新epsilon值.

        达到最小值不再衰减
        否则每次乘以衰减率，以更新epsilon值
        """
        if not self.epsilon_need_decay:
            return
        epsilon = self.epsilon * self.epsilon_decay
        if epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
            self.epsilon_need_decay = False
        else:
            self.epsilon = epsilon

    def _learn(self):
        # 从记忆库中采样训练数据
        b_state, b_action, b_reward, b_next_state, b_done = self.memory.sample(self.batch_size)
        # 转成tensor对象
        b_state = o2t(b_state)
        b_action = o2t(b_action, target_type=LongTensor)
        b_reward = o2t(b_reward)
        b_next_state = o2t(b_next_state)
        b_done = o2t(b_done)

        # 委托给model对象以更新网络参数
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
