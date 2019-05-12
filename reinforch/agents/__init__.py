"""
TODO
DQN                 v
DoubleDQN           v
DuelingDQN          v
Prioritize          v
Rainbow             v
PolicyGradient      v
AC                  x
A3C                 x
A2C                 x
DDPG                x
D4PG                x
"""
from reinforch.agents.agent import Agent
from reinforch.agents.dqn_agent import DQNAgent
from reinforch.agents.pg_agent import PolicyGradientAgent
from reinforch.agents.ac_agent import ActorCriticAgent

agents = dict(
    dqn=DQNAgent,
    pg=PolicyGradientAgent,
    ac=ActorCriticAgent,
)

__all__ = ['agents', 'Agent', 'DQNAgent', 'PolicyGradientAgent', 'ActorCriticAgent']
