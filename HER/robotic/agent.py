import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from networks import ActorNetwork, CriticNetwork
from normalizer import Normalizer
from utils import sync_networks, sync_grads


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, n_actions, action_space,
                 gamma=0.99, action_noise=0.05, explore=0.2, obs_shape=[8],
                 goal_shape=[3], max_size=1_000_000, fc1_dims=256,
                 fc2_dims=256, fc3_dims=256):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.action_space = action_space
        self.n_actions = n_actions
        self.limit = -1 / (1 - self.gamma)
        self.action_noise = action_noise * self.action_space.high
        self.explore = explore

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  fc3_dims, n_actions=n_actions,
                                  name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    fc3_dims, n_actions=n_actions,
                                    name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         fc3_dims, n_actions=n_actions,
                                         name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims,
                                           fc2_dims, fc3_dims,
                                           n_actions=n_actions,
                                           name='target_critic')

        self.noise = Normal(T.zeros(n_actions), T.tensor(self.action_noise))

        self.update_network_parameters(tau=1)

        sync_networks(self.actor)
        sync_networks(self.critic)

        self.obs_stats = Normalizer(obs_shape, 0.01, 5)
        self.goal_stats = Normalizer(goal_shape, 0.01, 5)

    def choose_action(self, observation, evaluate):
        if evaluate:
            with T.no_grad():
                state = T.tensor([observation],
                                 dtype=T.float).to(self.actor.device)
                _, pi = self.target_actor.forward(state)
                action = pi.cpu().detach().numpy().squeeze()
            return action
        if np.random.uniform() <= self.explore:
            action = self.action_space.sample()
        else:
            state = T.tensor([observation],
                             dtype=T.float).to(self.actor.device)
            _, pi = self.actor.forward(state)
            noise = self.noise.sample().to(self.actor.device)
            action = (pi + noise).cpu().detach().numpy().squeeze()
            action = np.clip(action, -1., 1.)
        return action

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, memories):
        states, actions, rewards, states_, done, goals = memories
        states = np.concatenate([states, goals], axis=1)
        states_ = np.concatenate([states_, goals], axis=1)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        _, target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        critic_value = critic_value.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(critic_value_.size(), 1)
        target = T.clamp(target, min=self.limit, max=0)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        mu, pi = self.actor.forward(states)
        actor_loss = self.critic.forward(states, pi)
        actor_loss = -T.mean(actor_loss)
        actor_loss += mu.pow(2).mean()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor.optimizer.step()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
