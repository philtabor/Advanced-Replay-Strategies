import numpy as np


class EpisodeWorker:
    def __init__(self, env, agent, memory):
        self.agent = agent
        self.env = env
        self.memory = memory
        self.get_slices()

    def get_slices(self):
        OB = self.env.observation_space['observation'].shape[0]
        A = self.env.observation_space['achieved_goal'].shape[0]
        D = self.env.observation_space['desired_goal'].shape[0]

        self.ob = slice(0, OB)
        self.ag = slice(OB, OB + A)
        self.dg = slice(OB + A, OB + A + D)

    def play_episode(self, evaluate=False):
        observation = self.env.reset()
        done = False
        score = 0
        desired_goal = observation[self.dg]
        achieved_goal = observation[self.ag]
        observation = observation[self.ob]

        self.agent.obs_stats.update_local_stats(observation)
        self.agent.goal_stats.update_local_stats(desired_goal)

        achieved_goal = self.agent.goal_stats.normalize_observation(
                achieved_goal)
        desired_goal = self.agent.goal_stats.normalize_observation(
                desired_goal)
        observation = self.agent.obs_stats.normalize_observation(observation)

        states, actions, rewards, states_,\
            dones, dg, ag, ag_ = [], [], [], [], [], [], [], []

        while not done:
            action = self.agent.choose_action(np.concatenate(
                               [observation, desired_goal]), evaluate)
            observation_, reward, done, info = self.env.step(action)

            achieved_goal_new = observation_[self.ag]
            observation_ = observation_[self.ob]

            if not evaluate:
                self.agent.goal_stats.update_local_stats(achieved_goal)
                self.agent.obs_stats.update_local_stats(observation_)

            observation_ = self.agent.obs_stats.normalize_observation(
                    observation_)
            achieved_goal_new = self.agent.goal_stats.normalize_observation(
                    achieved_goal_new)

            states.append(observation)
            states_.append(observation_)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            dg.append(desired_goal)
            ag.append(achieved_goal)
            ag_.append(achieved_goal_new)

            score += reward

            achieved_goal = achieved_goal_new
            observation = observation_

        if not evaluate:
            self.agent.obs_stats.recompute_global_stats()
            self.agent.goal_stats.recompute_global_stats()
            self.memory.store_episode([states, actions, rewards,
                                       states_, dones, dg, ag, ag_])
        success = info['is_success']
        return score, success
