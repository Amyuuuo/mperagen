class BasePolicy:
    def select_action(self, observation, env=None):
        raise NotImplementedError("This method should be overridden.")
    
    def select_action_multienv(self, observations, envs=None):
        """
        by default, select action for each environment independently
        Note that this could be optimized for parallel execution
        """
        if envs is None:
            envs = [None] * len(observations)
        return [self.select_action(obs, env) for obs, env in zip(observations, envs)]