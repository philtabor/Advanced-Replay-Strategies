import threading
import numpy as np
from mpi4py import MPI


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.local_sum = np.zeros(self.size, dtype=np.float32)
        self.local_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.local_cnt = np.zeros(1, dtype=np.float32)

        self.lock = threading.Lock()

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_cnt = 1

    def update_local_stats(self, new_data):
        with self.lock:
            self.local_sum += new_data.sum(axis=0)
            self.local_sum_sq += (np.square(new_data)).sum(axis=0)
            self.local_cnt[0] += new_data.shape[0]

    def sync_thread_stats(self, local_sum, local_sum_sq, local_cnt):
        local_sum[...] = self.mpi_average(local_sum)
        local_sum_sq[...] = self.mpi_average(local_sum_sq)
        local_cnt[...] = self.mpi_average(local_cnt)
        return local_sum, local_sum_sq, local_cnt

    def mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    def normalize_observation(self, v):
        clip_range = self.default_clip_range
        return np.clip((v - self.running_mean) / self.running_std,
                       -clip_range, clip_range).astype(np.float32)

    def recompute_global_stats(self):
        with self.lock:
            local_cnt = self.local_cnt.copy()
            local_sum = self.local_sum.copy()
            local_sum_sq = self.local_sum_sq.copy()

            self.local_cnt[...] = 0
            self.local_sum[...] = 0
            self.local_sum_sq[...] = 0

        sync_sum, sync_sum_sq, sync_cnt = self.sync_thread_stats(
                local_sum, local_sum_sq, local_cnt)

        self.running_cnt += sync_cnt
        self.running_sum += sync_sum
        self.running_sum_sq += sync_sum_sq

        self.running_mean = self.running_sum / self.running_cnt
        tmp = self.running_sum_sq / self.running_cnt -\
            np.square(self.running_sum / self.running_cnt)
        self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))
