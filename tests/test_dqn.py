from reinforch.agents import DQNAgent
from reinforch.core.memorys import SimpleMatrixMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner


def test_dqn():
    gym_id = 'CartPole-v0'

    env = OpenAIGym(gym_id)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    memory = SimpleMatrixMemory(row_size=3000, every_class_size=[n_s, 1, 1, n_s, 1])
    agent = DQNAgent(n_s=n_s,
                     n_a=n_a,
                     memory=memory,
                     config='tests/configs/test_dqn.json')
    with Runner(agent=agent,
                environment=env,
                verbose=False) as runner:
        runner.train(total_episode=10,
                     max_step_in_one_episode=200,
                     save_model=False,
                     save_final_model=False,
                     visualize=False)

