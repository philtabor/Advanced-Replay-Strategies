import numpy as np


class HER:
    def __init__(self, max_mem, input_shape, n_actions, goal_shape, batch_size,
                 reward_fn, strategy='final'):
        self.max_mem = max_mem
        self.strategy = strategy
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.reward_fn = reward_fn

        self.states = np.zeros((max_mem, input_shape),
                               dtype=np.float64)
        self.states_ = np.zeros((max_mem, input_shape),
                                dtype=np.float64)
        self.actions = np.zeros((max_mem, n_actions),
                                dtype=np.float32)
        self.rewards = np.zeros(max_mem, dtype=np.float32)
        self.dones = np.zeros(max_mem, dtype=np.bool)
        self.desired_goals = np.zeros((max_mem, goal_shape), dtype=np.float64)
        self.achieved_goals = np.zeros((max_mem, goal_shape), dtype=np.float64)
        self.achieved_goals_ = np.zeros((max_mem, goal_shape),
                                        dtype=np.float64)

    def store_memory(self, state, action, reward, state_, done,
                     d_goal, a_goal, a_goal_):
        index = self.mem_cntr % self.max_mem
        self.states[index] = state
        self.states_[index] = state_
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.desired_goals[index] = d_goal
        self.achieved_goals[index] = a_goal
        self.achieved_goals_[index] = a_goal_
        self.mem_cntr += 1

    def store_episode(self, ep_memory):
        states, actions, rewards, states_, dones, dg, ag, ag_ = ep_memory

        if self.strategy == 'final':
            hindsight_goals = [[ag_[-1]]] * len(ag_)

        elif self.strategy is None:
            hindsight_goals = [[dg[0]]] * len(dg)

        for idx, s in enumerate(states):
            self.store_memory(s, actions[idx], rewards[idx], states_[idx],
                              dones[idx], dg[idx], ag[idx], ag_[idx])
            for goal in hindsight_goals[idx]:
                reward = self.reward_fn(ag_[idx], goal, {})
                self.store_memory(s, actions[idx], reward, states_[idx],
                                  dones[idx], goal, ag[idx], ag_[idx])

    def sample_memory(self):
        last_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(last_mem, self.batch_size, replace=False)

        return self.states[batch], self.actions[batch], self.rewards[batch],\
            self.states_[batch], self.dones[batch],\
            self.desired_goals[batch]

    def ready(self):
        return self.mem_cntr > self.batch_size
