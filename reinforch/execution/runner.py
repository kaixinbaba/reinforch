import os
import time

from tqdm import tqdm

from reinforch.agents import Agent
from reinforch.core.logger import Log, INFO, DEBUG
from reinforch.environments import Environment
from reinforch.exception import ReinforchException


class Runner(object):
    """
    整个RL算法的执行器，通常是在入口文件中使用

    """

    def __init__(self,
                 agent: Agent,
                 environment: Environment,
                 save_dest_folder: str = '.',
                 exists_model: str = None,
                 overwritten: bool = True,
                 verbose=False):
        self.log = Log(__name__, level=DEBUG) if verbose else Log(__name__, level=INFO)
        self.agent = agent
        self.environment = environment
        self.overwritten = overwritten
        self.save_file_name = '{}_{}_{{}}.pkl'.format(str(self.agent), str(self.environment))
        if not os.path.exists(save_dest_folder):
            # 不存在就创建
            os.mkdir(save_dest_folder)
        elif not os.path.isdir(save_dest_folder):
            # 传入的不是目录，使用当前目录
            save_dest_folder = '.'
        self.save_dest_folder = save_dest_folder
        self.save_path = os.path.join(self.save_dest_folder, self.save_file_name)
        if exists_model is not None:
            if os.path.exists(exists_model) and os.path.isfile(exists_model):
                self.agent.load(exists_model)
            else:
                self.log.warn('specified path [{}] is incorrect model file'.format(exists_model))
        self.current_episode = 1
        self.reset()

    def reset(self):
        pass

    def train(self,
              total_episode: int = 10000,
              max_step_in_one_episode: int = None,
              save_episode: int = 1000,
              save_final_model: bool = True,
              visualize=False):
        self.environment.visualize = visualize
        if save_episode is None or save_episode > total_episode:
            if total_episode > 10:
                # 默认以10%的频率的保存
                save_episode = total_episode // 10
            else:
                # 小于10episode的每一步都保存
                save_episode = 1
        self.log.info(
            'Start play! Agent : {agent}, Environment : {environment}'.format(
                agent=self.agent,
                environment=self.environment,
            ))
        self.log.info(
            'With total [{total_episode}] episode, max step count in one episode [{max_step_in_one_episode}]'.format(
                total_episode=total_episode,
                max_step_in_one_episode=max_step_in_one_episode,
            ))
        for episode in tqdm(range(1, total_episode + 1), ncols=100):
            episode_start_time = time.time()
            state = self.environment.reset()
            episode_reward = 0
            step = 0
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.execute(action=action)
                episode_reward += reward
                self.agent.step(state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                done=done,
                                info=info)
                if episode % save_episode == 0:
                    # save model
                    self.__save(self.save_path.format(episode))
                if done or (max_step_in_one_episode is not None and step == max_step_in_one_episode):
                    break
                state = next_state
                step += 1
            if episode == total_episode and save_final_model:
                # save final model
                self.__save(self.save_path.format('last'))
            cost_time = time.time() - episode_start_time
            self.log.debug('[train] {} episode, reward : {}, cost time : {}'.format(episode, episode_reward, cost_time))

    def test(self, total_episode: int = 5, max_step_in_one_episode: int = None, suffix='last', visualize=True):
        self.environment.visualize = visualize
        self.__load(suffix)
        for episode in tqdm(range(1, total_episode + 1), ncols=100):
            episode_start_time = time.time()
            state = self.environment.reset()
            episode_reward = 0
            step = 0
            while True:
                action = self.agent.act(state)
                self.environment.execute(action)
                next_state, reward, done, info = self.environment.execute(action=action)
                if done or (max_step_in_one_episode is not None and step == max_step_in_one_episode):
                    break
                step += 1
                state = next_state

            cost_time = time.time() - episode_start_time
            self.log.debug('[test] {} episode, reward : {}, cost time : {}'.format(episode, episode_reward, cost_time))

    def __save(self, path):
        if not os.path.exists(path) or not self.overwritten:
            self.agent.save(path)

    def __load(self, suffix='last'):
        _path = self.save_path.format(suffix)
        if not os.path.exists(_path):
            raise ReinforchException('"{}" does not exists!'.format(_path))
        self.agent.load(_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.agent.close()
        self.environment.close()
