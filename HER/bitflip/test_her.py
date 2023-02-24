import numpy as np
import random
from bit_flip import BitFlipEnv
from her import HER

n_bits = 16
env = BitFlipEnv(n_bits)
random.seed(123)
np.random.seed(123)
batch_size = 2
max_size = 1_000
input_shape = n_bits
memory = HER(max_mem=max_size, input_shape=input_shape, n_actions=1,
             batch_size=batch_size, goal_shape=n_bits,
             strategy='final', reward_fn=env.compute_reward)

for _ in range(40):
    o = env.reset()
    agl = o[n_bits:2*n_bits]
    dg_ = o[2*n_bits:3*n_bits]
    d = False
    s, a, re, dn, s_, dg, ag, ag_ = [], [], [], [], [], [], [], []
    while not d:
        action = env.action_space_sample()
        o_, r, d, i = env.step(action)
        agl_ = o_[n_bits:2*n_bits]
        s.append(o[:n_bits])
        a.append(action)
        re.append(r)
        dn.append(d)
        s_.append(o_[:n_bits])
        dg.append(dg_)
        ag.append(agl)
        ag_.append(agl_)
        agl = agl_
        o = o_
    memory.store_episode([s, a, re, s_, dn, dg, ag, ag_])
    assert memory.ready(), 'Unexpected number of memories in buffer'

s, a, re, s_, dn, dg, ag = memory.sample_memory()

data = np.load('results.npy')

assert (s[0] == data[0]).all(), 'Unexpected values for sampling of states'
assert (s[1] == data[1]).all(), 'Unexpected values for sampling of states'
assert (s_[0] == data[2]).all(), 'Unexpected values for sampling of states_'
assert (s_[1] == data[3]).all(), 'Unexpected values for sampling of states_'
