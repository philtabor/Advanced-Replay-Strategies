import numpy as np


def max_heapify(array, i, N=None):
    N = len(array) if N is None else N
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i

    if left < N and array[left] > array[i]:
        largest = left
    if right < N and array[right] > array[largest]:
        largest = right
    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        max_heapify(array, largest, N)
    return array


def build_max_heap(array):
    N = len(array)
    for i in range(N//2, -1, -1):
        array = max_heapify(array, i)
    return array


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
