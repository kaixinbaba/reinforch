from reinforch.agents import DQNAgent
from reinforch.core.memorys import SimpleMatrixMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner

if __name__ == '__main__':

    def cartpole_reward_shape(state=None, reward=None, terminal=None, env=None):
        x, x_dot, theta, theta_dot = state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

    env = OpenAIGym('CartPole-v0', visualize=False, reward_shape=cartpole_reward_shape)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    memory = SimpleMatrixMemory(row_size=2000, every_class_size=[n_s, 1, 1, n_s, 1])
    agent = DQNAgent(n_s=n_s,
                     n_a=n_a,
                     memory=memory,
                     lr=0.001,
                     gamma=0.99,
                     c_step=50,
                     soft_update=False,
                     tau=0.01,
                     epsilon=1.0,
                     epsilon_decay=0.99,
                     epsilon_min=0.0,
                     batch_size=32,
                     learn_threshold=None,
                     action_dim=None,
                     double_dqn=True,
                     dueling_dqn=True,
                     config='configs/openai_gym_CartPole.json')
    with Runner(agent=agent,
                environment=env,
                total_episode=500,
                max_step_in_one_episode=1000) as runner:
        runner.train()
