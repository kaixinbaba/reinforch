class Environment(object):
    """
    Base environment class.
    """

    def __str__(self):
        raise NotImplementedError

    def close(self):
        """
        Clean up resouces, After close this environment should not be called.
        """
        pass

    def seed(self, seed=None):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return seed

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        raise NotImplementedError

    def execute(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            :param action: Actions to execute.

        Returns:
            Tuple of (next state, reward, bool indicating terminal, other info)
        """
        raise NotImplementedError

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are
        available simultaneously.

        Returns:
            States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
        """
        raise NotImplementedError

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are
        available simultaneously.

        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
        """
        raise NotImplementedError
