"""
TODO
DQN                 v
DoubleDQN           v
DuelingDQN          v
Prioritize          x
Rainbow             x
PolicyGradient      x
A3C                 x
A2C                 x
DDPG                x
D4PG                x
"""
from reinforch.agents.agent import Agent, DQNAgent

agents = dict(
    dqn=DQNAgent,
)

__all__ = ['agents', 'Agent', 'DQNAgent']
