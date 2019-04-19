from reinforch.agents import DQNAgent
from reinforch.core.logger import Log, INFO
from reinforch.core.memorys import PrioritizeMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner

logger = Log(__name__, level=INFO)

if __name__ == '__main__':
    def reward_shape(state=None, reward=None, terminal=None, env=None):
        position, velocity = state
        reward = abs(position - (-0.5))
        return reward


    def reward_shape2(state=None, reward=None, terminal=None, env=None):
        if terminal:
            reward = 10
        return reward


    env = OpenAIGym('MountainCar-v0', reward_shape=reward_shape2)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    memory = PrioritizeMemory(capacity=2000,
                              every_class_size=[n_s, 1, 1, n_s, 1])

    agent = DQNAgent(n_s=n_s,
                     n_a=n_a,
                     memory=memory,
                     config='configs/openai_gym_MountainCar.json')
    with Runner(agent=agent,
                environment=env,
                save_dest_folder='gym_mountaincar_save_point',
                verbose=True) as runner:
        runner.train(total_episode=5,
                     save_final_model=True,
                     visualize=False)

        logger.info('The agent has completed its training...')

        runner.test(visualize=True)
