from reinforch.agents import Agent
from reinforch.environments import Environment


class Runner(object):

    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment
        self.reset()

    def reset(self):
        self.environment.reset()

    def run(self):
        pass

    def close(self):
        self.agent.close()
        self.environment.close()
