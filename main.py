import sys

from em_improved_version import EMImprovedVersion
from utils import read_image, plot, error, create_image
import numpy as np


def true_data():
    K = [1, 2, 4, 8, 16, 32, 64]
    path = "im.jpg"

    for k in K:
        im, shape, mode = read_image(path)

        em = EMImprovedVersion(im, k)

        Ls = em.EM_algorithm()

        final_im = create_image(em.G, em.M)
        show_image(final_im, shape, mode)

        err = error(im, final_im, em.N)
        print(err)


def test_data():
    N = 2000
    X = np.zeros((N, 2))
    P = [0.3, 0.6, 1]

    Mtrue = np.array([[0, 0], [-2, 3], [2, 3]])

    # generate X data
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
    # plot(X, Minit)
    # true data
    plot(X, Mtrue)

    em = EMImprovedVersion(X, K, True, True, np.array(P), Minit)

    em.EM_algorithm()

    plot(X, em.M)


if __name__ == "__main__":
    case = sys.argv[1]

    if case == 'true':
        true_data()
    elif case == 'test':
        test_data()
