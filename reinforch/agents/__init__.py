from reinforch.agents.agent import Agent, DQNAgent

agents = dict(
    dqn=DQNAgent,
)

__all__ = ['agents', 'Agent', 'DQNAgent']
