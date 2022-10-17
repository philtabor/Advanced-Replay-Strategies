import numpy as np
import torch as T
from network import LinearDeepQNetwork
from memory import SumTree


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, alpha=0.5, beta=0, r_iter=32,
                 algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.rebalance_iter = r_iter

        self.memory = SumTree(mem_size, batch_size, alpha=alpha, beta=beta)

        self.q_eval = LinearDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=self.env_name+'_q_eval',
                                         chkpt_dir=self.chkpt_dir)

        self.q_next = LinearDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=self.env_name+'_q_next',
                                         chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],
                             dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition([state, action, reward, state_, done])

    def sample_memory(self):
        sarsd, sample_idx, weights = self.memory.sample()

        states = np.array([row[0] for row in sarsd])
        actions = np.array([row[1] for row in sarsd])
        rewards = np.array([row[2] for row in sarsd])
        states_ = np.array([row[3] for row in sarsd])
        dones = np.array([row[4] for row in sarsd])

        states = T.tensor(states, dtype=T.float).to(self.q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.q_eval.device)

        weights = T.tensor(weights, dtype=T.float).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, sample_idx, weights

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if not self.memory.ready():
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones,\
            sample_idx, weights = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        td_error = np.abs((q_target.detach().cpu().numpy() -
                          q_pred.detach().cpu().numpy()))
        td_error = np.clip(td_error, 0., 1.)

        self.memory.update_priorities(sample_idx, td_error)

        q_target *= weights
        q_pred *= weights

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
