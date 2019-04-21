from reinforch.agents import PolicyGradientAgent
from reinforch.core.logger import Log, INFO
from reinforch.core.memorys import PGMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner
from examples import default_config_path, default_save_folder

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

    gym_id = 'MountainCarContinuous-v0'

    env = OpenAIGym(gym_id, reward_shape=reward_shape2)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    action_dim = env.actions.get('max_value')
    memory = PGMemory()

    agent = PolicyGradientAgent(n_s=n_s,
                                n_a=n_a,
                                action_dim=action_dim,
                                memory=memory,
                                config=default_config_path('pg', gym_id))
    with Runner(agent=agent,
                environment=env,
                save_dest_folder=default_save_folder('pg', gym_id),
                verbose=False) as runner:
        runner.train(total_episode=5,
                     save_final_model=True,
                     visualize=False)

        logger.info('The agent has completed its training...')

        runner.test(visualize=True)
