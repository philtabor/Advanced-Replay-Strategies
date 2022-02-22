from mpi4py import MPI
import numpy as np
import torch as T


def sync_networks(network):
    comm = MPI.COMM_WORLD

    params = np.concatenate([getattr(p, 'data').cpu().numpy().flatten()
                             for p in network.parameters()])
    comm.Bcast(params)
    idx = 0
    for p in network.parameters():
        getattr(p, 'data').copy_(T.tensor(
            params[idx:idx + p.data.numel()]).view_as(p.data))
        idx += p.data.numel()


def sync_grads(network):
    comm = MPI.COMM_WORLD

    grads = np.concatenate([getattr(p, 'grad').cpu().numpy().flatten()
                           for p in network.parameters()])
    global_grads = np.zeros_like(grads)
    comm.Allreduce(grads, global_grads, op=MPI.SUM)
    idx = 0
    for p in network.parameters():
        getattr(p, 'grad').copy_(T.tensor(
            global_grads[idx:idx + p.data.numel()]).view_as(p.data))
        idx += p.data.numel()
