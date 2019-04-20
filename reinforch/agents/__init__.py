"""
TODO
DQN                 v
DoubleDQN           v
DuelingDQN          v
Prioritize          v
Rainbow             x
PolicyGradient      x
A3C                 x
A2C                 x
DDPG                x
D4PG                x
"""
from reinforch.agents.agent import Agent
from reinforch.agents.dqn_agent import DQNAgent

agents = dict(
    dqn=DQNAgent,
)

__all__ = ['agents', 'Agent', 'DQNAgent']
