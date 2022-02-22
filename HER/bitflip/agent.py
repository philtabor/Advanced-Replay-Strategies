import numpy as np
import torch as T
from networks import DeepQNetwork


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo='dqn', env_name='bit_flip',
                 chkpt_dir='models'):
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

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_next',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, evaluate=False):
        if np.random.random() > self.epsilon or evaluate:
            state = T.tensor([observation],
                             dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

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

    def learn(self, memories):
        state, action, reward, new_state, done, dg = memories
        state = np.concatenate([state, dg], axis=1)
        new_state = np.concatenate([new_state, dg], axis=1)

        states = T.tensor(state, dtype=T.float).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action, dtype=T.long).to(self.q_eval.device)
        states_ = T.tensor(new_state, dtype=T.float).to(self.q_eval.device)

        actions = actions.view(-1)

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
