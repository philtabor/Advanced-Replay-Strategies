import numpy as np


def max_heapify(array, i, N=None):
    pass


def build_max_heap(array):
    pass


if __name__ == '__main__':
    np.random.seed(42)
    a = np.random.choice(np.arange(100), 21, replace=False)
    print('unsorted array: {}'.format(a))
    a = build_max_heap(a)
    reference = np.array([90., 80., 83., 77., 55., 73., 70., 76.,
                          53., 44., 18., 30., 39., 33., 22., 4.,
                          45., 10., 12., 31., 0])
    print('max heap array: {}'.format(a))
    assert (a == reference).all()
