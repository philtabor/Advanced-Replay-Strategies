import random
import numpy as np
import torch as T
from agent import Agent
from episode import EpisodeWorker
from her import HER
from bit_flip import BitFlipEnv


def train(agent, worker, memory):
    epochs = 100
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
            # cycle_avg_score = np.mean(score_history)
            # cycle_avg_success = np.mean(success_history)
            # print('Epoch: {} Cycle: {} Training Avg Score {:.1f} '
            #      'Trainig Avg Success: {:.3f}'.
            #      format(epoch, cycle, cycle_avg_score, cycle_avg_success))
            if memory.ready():
                for _ in range(n_updates):
                    memories = memory.sample_memory()
                    agent.learn(memories)
        score_history, success_history = [], []
        for episode in range(n_tests):
            score, success = worker.play_episode(evaluate=True)
            success_history.append(success)
            score_history.append(score)
        avg_success = np.mean(success_history)
        avg_score = np.mean(score_history)
        print('Epoch: {} Testing Agent. Avg Score: {:.1f} '
              'Avg Sucess: {:.3f}'.
              format(epoch, avg_score, avg_success))


def main():
    n_bits = 32
    env = BitFlipEnv(n_bits, max_steps=n_bits)
    random.seed(123)
    np.random.seed(123)
    T.manual_seed(123)
    T.cuda.manual_seed(123)

    batch_size = 128
    max_size = 1_000_000
    input_shape = n_bits
    memory = HER(max_mem=max_size, input_shape=input_shape, n_actions=1,
                 batch_size=batch_size, goal_shape=n_bits, strategy=None,
                 reward_fn=env.compute_reward)
    agent = Agent(lr=0.001, epsilon=0.2, n_actions=n_bits, eps_dec=0.0,
                  batch_size=batch_size, input_dims=2*input_shape, gamma=0.98)
    ep_worker = EpisodeWorker(env, agent, memory)

    train(agent, ep_worker, memory)


if __name__ == '__main__':
    main()
