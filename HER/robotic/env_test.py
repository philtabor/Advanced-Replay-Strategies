import panda_gym
import gym
import time

env = gym.make('PandaReach-v2', render=True)

for _ in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        time.sleep(0.05)
env.close()
