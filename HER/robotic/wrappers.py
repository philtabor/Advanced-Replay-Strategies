import numpy as np
import gym


class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenDictWrapper, self).__init__(env)

    def observation(self, obs):
        obs = np.concatenate([obs[k] for k in obs.keys()])
        return obs
