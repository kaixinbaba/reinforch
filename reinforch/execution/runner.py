from tqdm import tqdm

from reinforch.agents import Agent
from reinforch.core.logger import Log, INFO, DEBUG
from reinforch.environments import Environment

import time


class Runner(object):

    def __init__(self,
                 agent: Agent,
                 environment: Environment,
                 total_episode=10000,
                 max_step_in_one_episode=1000,
                 save_episode=1000,
                 verbose=False):
        self.agent = agent
        self.environment = environment
        self.total_episode = total_episode
        self.max_step_in_one_episode = max_step_in_one_episode
        self.save_episode = save_episode
        self.current_episode = 1
        self.log = Log(__name__, level=DEBUG) if verbose else Log(__name__, level=INFO)
        self.reset()

    def reset(self):
        pass

    def run(self):
        self.log.info(
            'Start play! Agent : {agent}, Environment : {environment}'.format(
                agent=self.agent,
                environment=self.environment,
            ))
        self.log.info(
            'With total {total_episode} episode, max step count in one episode {max_step_in_one_episode}'.format(
                total_episode=self.total_episode,
                max_step_in_one_episode=self.max_step_in_one_episode,
            ))
        for episode in range(1, self.total_episode + 1):
            episode_start_time = time.time()
            # self.log.debug('Current episode {episode}')
            state = self.environment.reset()
            for step in tqdm(range(self.max_step_in_one_episode), ncols=120):
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.execute(action=action)
                self.agent.step(state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                done=done,
                                info=info)
                if done:
                    break
                state = next_state
            cost_time = time.time() - episode_start_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.agent.close()
        self.environment.close()
