import numpy as np


def likelihood(p, s, m, x):
    # p: K
    # s: K
    # x: D
    # m: K*D

    # output: 1

    K = m.shape[0]
    p1 = 1 / np.sqrt(2 * np.pi * np.power(s))  # K
    copy_K_times = np.tile(x, (K, 1))  # K * 3
    sub = copy_K_times - m  # K * 3
    p2 = - np.power(sub) / (2 * np.power(s))
    p3 = np.exp(p2)
    p4 = np.prod(p1 * p3)  # K

    return np.sum(p * p4)
