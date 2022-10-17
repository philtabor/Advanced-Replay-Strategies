import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve


def clip_reward(r):
    if r > 1:
        return 1
    elif r < -1:
        return -1
    else:
        return r


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 1000
    alpha = 0.6
    beta = 0.4
    replace = 250
    bs = 64
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-4, alpha=alpha,
                  beta=beta, input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50*1024, eps_min=0.01,
                  batch_size=bs, replace=replace, eps_dec=1e-4,
                  chkpt_dir='models/', algo='ddqn', env_name='CartPole-v0')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
        + str(n_games) + 'games' + str(alpha) +\
        'alpha_' + str(beta) + '_replace_' + str(replace)
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing,
    # do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    # env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            r = clip_reward(reward)
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       r, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode {} score {:.1f} eps {:.2f} n steps {}'.
              format(i, avg_score, agent.epsilon, n_steps))

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
