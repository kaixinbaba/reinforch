from reinforch.agents import DQNAgent
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner
from reinforch.core.memorys import SimpleMatrixMemory

if __name__ == '__main__':
    env = OpenAIGym('CartPole-v0', visualize=True)
    n_s = env.n_s
    n_a = env.n_a
    memory = SimpleMatrixMemory(row_size=2000, every_class_size=[n_s, 1, 1, n_s, 1])
    agent = DQNAgent(n_s=n_s, n_a=n_a, memory=memory, config='configs/openai_gym_CartPole.json')
    with Runner(agent=agent, environment=env) as runner:
        runner.train()
