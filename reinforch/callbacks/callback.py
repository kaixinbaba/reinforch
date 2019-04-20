from typing import Union, List, Tuple


class CallBack(object):

    def __init__(self):
        pass

    def episode_begin(self, episode):
        pass

    def after_environment_reset(self, episode, state):
        pass

    def before_agent_action(self, step, state):
        pass

    def after_agent_action(self, step, state, action):
        pass

    def after_environment_execute(self, step, state, action, reward, next_action, done, info=None):
        pass

    def episode_end(self, episode, episode_reward, cost_time):
        pass


class LogCallBack(CallBack):

    def __init__(self, logger):
        super(LogCallBack, self).__init__()
        self.logger = logger

    def episode_begin(self, episode):
        self.logger.debug('>> episode {} get started...'.format(episode))

    def before_agent_action(self, step, state):
        self.logger.debug('>>>> step {}, state : {}'.format(step, state))

    def after_agent_action(self, step, state, action):
        self.logger.debug('>>>> step {}, agent choose action {}'.format(step, action))

    def after_environment_execute(self, step, state, action, reward, next_state, done, info=None):
        self.logger.debug('>>>> step {}, environment return next_state {}, reward {}, done {}, info {}'.format(
            step, next_state, reward, done, info
        ))

    def episode_end(self, episode, episode_reward, cost_time):
        self.logger.debug('>> episode {} end, total reward {}, cost time {}'.format(
            episode, episode_reward, cost_time
        ))


class CallBackList(CallBack):

    def __init__(self, callbacks=None, logger=None):
        super(CallBackList, self).__init__()
        self.callbacks = [] if callbacks is None else callbacks
        # default log callback
        self.callbacks.append(LogCallBack(logger=logger))

    def register_callback(self, callbacks: Union[CallBack, List[CallBack], Tuple[CallBack]]):
        if isinstance(callbacks, (list, tuple)) and all(map(lambda c: isinstance(c, CallBack), callbacks)):
            self.callbacks += callbacks
        elif isinstance(callbacks, CallBack):
            self.callbacks.append(callbacks)
        else:
            raise TypeError('type must be Callback')

    def episode_begin(self, episode):
        for callback in self.callbacks:
            callback.episode_begin(episode)

    def after_environment_reset(self, episode, state):
        for callback in self.callbacks:
            callback.after_environment_reset(episode, state)

    def before_agent_action(self, step, state):
        for callback in self.callbacks:
            callback.before_agent_action(step, state)

    def after_agent_action(self, step, state, action):
        for callback in self.callbacks:
            callback.after_agent_action(step, state, action)

    def after_environment_execute(self, step, state, action, reward, next_action, done, info=None):
        for callback in self.callbacks:
            callback.after_environment_execute(step, state, action, reward, next_action, done, info)

    def episode_end(self, episode, episode_reward, cost_time):
        for callback in self.callbacks:
            callback.episode_end(episode, episode_reward, cost_time)
