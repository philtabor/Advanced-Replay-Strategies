from bit_flip import BitFlipEnv


if __name__ == '__main__':
    env = BitFlipEnv(n_bits=4)

    for _ in range(2):
        done = False
        obs = env.reset()
        print('starting new episode')
        while not done:
            action = env.action_space_sample()
            obs_, reward, done, info = env.step(action)
            env.render()
