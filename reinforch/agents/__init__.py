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
from reinforch.agents.pg_agent import PolicyGradientAgent

agents = dict(
    dqn=DQNAgent,
    pg=PolicyGradientAgent,
)

__all__ = ['agents', 'Agent', 'DQNAgent', 'PolicyGradientAgent']
