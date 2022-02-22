import numpy as np


class BitFlipEnv:
    def __init__(self, n_bits, max_steps=50):
        self.n_bits = n_bits
        self.max_steps = max_steps
        self.n_actions = n_bits

        self.observation = self.reset()
        self.observation_space = {'observation': np.empty((self.n_bits)),
                                  'achieved_goal': np.empty((self.n_bits)),
                                  'desired_goal': np.empty((self.n_bits)),
                                  }

    def reset(self):
        self._bits = np.array([np.random.randint(2)
                              for _ in range(self.n_bits)])
        self._desired_goal = np.array([np.random.randint(2)
                                      for _ in range(self.n_bits)])
        self._achieved_goal = self._bits.copy()

        obs = np.concatenate([self._bits,
                              self._achieved_goal,
                              self._desired_goal])
        self._step = 0
        return obs

    def compute_reward(self, desired_goal, achieved_goal, info):
        reward = 0.0 if (desired_goal == achieved_goal).all() else -1.0
        return reward

    def step(self, action):
        assert action <= self.n_actions, "Invalid Action"
        new_bit = 0 if self._bits[action] == 1 else 1
        self._bits[action] = new_bit
        info = {}
        self._achieved_goal = self._bits.copy()
        reward = self.compute_reward(self._desired_goal,
                                     self._achieved_goal, {})
        self._step += 1
        if reward == 0.0 or self._step >= self.max_steps:
            done = True
        else:
            done = False
        info['is_success'] = 1.0 if reward == 0.0 else 0.0
        obs = np.concatenate([self._bits, self._achieved_goal,
                              self._desired_goal])
        return obs, reward, done, info

    def action_space_sample(self):
        return np.random.randint(0, self.n_actions)

    def render(self):
        for bit in self._bits:
            print(bit, end='  ')
        print('\n')
