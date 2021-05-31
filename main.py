import sys

from em_slow_version import EMSlowVersion
from utils import read_image, plot, plot_clusters
import numpy as np

def true_data():
    K = [1, 2, 4, 8, 16, 32, 64]

    im, width, height = read_image()

    em_slow_version = EMSlowVersion(im, K[3])

    em_slow_version.EM_algorithm()

    em_slow_version.show(width, height)

def test_data():
    N = 2000
    X = np.zeros((N, 2))
    P = [0.3, 0.6, 1]
    Mtrue = np.array([[0, 0], [-2, 3], [2, 3]])

    # genetate X data
    for i in range(N):
        u = np.random.rand()
        if u < P[0]:
            k = 0
        elif u >= P[0] and u < P[1]:
            k = 1
        else:
            k = 2
        X[i, :] = Mtrue[k] + 0.5 * np.random.randn(2)

    N, D = X.shape
    K = 3

    # initializa M
    Minit = np.array([np.mean(X, axis=0), ] * K) + np.random.randn(K, D)

    # initial data
    plot(X, Minit)
    # true data
    plot(X, Mtrue)

    em_slow_version = EMSlowVersion(X, K, True, True, P, Minit)

    em_slow_version.EM_algorithm()


if __name__ == "__main__":
    case = sys.argv[1]

    if case == 'true':
        true_data()
    elif case == 'test':
        test_data()