import os
import random
import gym
import panda_gym
from mpi4py import MPI
import numpy as np
import torch as T
from agent import Agent
from episode import EpisodeWorker
from her import HER
from wrappers import FlattenDictWrapper


def train(agent, worker, memory, environ):
    epochs = 50
    cycle_length = 16
    n_cycles = 50
    n_updates = 40
    n_tests = 10
    for epoch in range(epochs):
        for cycle in range(n_cycles):
            score_history, success_history = [], []
            for i in range(cycle_length):
                score, success = worker.play_episode()
                score_history.append(score)
                success_history.append(success)
            """
            if MPI.COMM_WORLD.Get_rank() == 0:
                cycle_avg_score = np.mean(score_history)
                cycle_avg_success = np.mean(success_history)

                print('Epoch: {} Cycle: {} Training Avg Score {:.1f} '
                      'Training Avg Success: {:.3f}'.
                      format(epoch, cycle, cycle_avg_score, cycle_avg_success))
            """
            if memory.ready():
                for _ in range(n_updates):
                    memories = memory.sample_memory()
                    agent.learn(memories)
                agent.update_network_parameters()
        score_history, success_history = [], []
        for episode in range(n_tests):
            score, success = worker.play_episode(evaluate=True)
            success_history.append(success)
            score_history.append(score)
        avg_success = np.mean(success_history)
        avg_score = np.mean(score_history)
        global_success = MPI.COMM_WORLD.allreduce(avg_success, op=MPI.SUM)
        global_score = MPI.COMM_WORLD.allreduce(avg_score, op=MPI.SUM)
        eval_score = global_score / MPI.COMM_WORLD.Get_size()
        eval_success = global_success / MPI.COMM_WORLD.Get_size()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Epoch: {} Testing Agent. Avg Score: {:.1f} '
                  'Avg Sucess: {:.3f} Environment: {}'.
                  format(epoch, eval_score, eval_success, environ))


def main():
    env_string = 'PandaPickAndPlace-v2'
    # env_string = 'PandaPush-v2'
    env = gym.make(env_string)
    env = FlattenDictWrapper(env)
    seed = 123 + MPI.COMM_WORLD.Get_rank()

    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)

    batch_size = 256
    max_size = 1_000_000
    obs_shape = env.observation_space['observation'].shape[0]
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    input_shape = obs_shape
    memory = HER(max_mem=max_size, input_shape=input_shape,
                 n_actions=env.action_space.shape[0],
                 batch_size=batch_size, goal_shape=goal_shape,
                 strategy='future', reward_fn=env.compute_reward)
    input_shape = obs_shape + goal_shape
    agent = Agent(alpha=0.001, beta=0.001, action_space=env.action_space,
                  input_dims=input_shape, tau=0.05, gamma=0.98,
                  fc1_dims=256, fc2_dims=256, fc3_dims=256,
                  n_actions=env.action_space.shape[0], explore=0.3,
                  obs_shape=obs_shape, goal_shape=goal_shape,
                  action_noise=0.2)
    ep_worker = EpisodeWorker(env, agent, memory)

    train(agent, ep_worker, memory, env_string)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main()
