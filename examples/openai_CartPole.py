from reinforch.agents import DQNAgent
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner

if __name__ == '__main__':
    env = OpenAIGym('CartPole-v0', visualize=True)
    agent = DQNAgent(n_s=env.states, n_a=env.actions, config='configs/openai_CartPole.json')
    with Runner(agent=agent, environment=env) as runner:
        runner.run()
