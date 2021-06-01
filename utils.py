from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames as mcolors


class MyImage():
    def __init__(self, im):
        self._im_np, self.shape, self.mode, self.avg, self.std = self.process_image(im)

    def get_image(self):
        return self._im_np

    def process_image(self, im):
        width, height = im.size
        mode = im.mode

        im_np = np.array(im.getdata())

        avg = np.average(im_np, axis=0)
        std = np.std(im_np, axis=0)

        im_np = (im_np - avg) / std
        shape = (height, width, im_np.shape[1])

        return im_np, shape, mode, avg, std


def create_image(G, M):
    return M[np.argmax(G, axis=1), :]


def get_image(final_im, im):
    return Image.fromarray(np.uint8(final_im.reshape(im.shape) * im.std + im.avg), im.mode)


def init(K, D):
    # P,M ,S
    return np.random.rand(K), np.random.randn(K, D), np.random.uniform(low=0.1, high=10, size=(K,))


def initS(K):
    return np.random.uniform(low=0.1, high=10, size=(K,))


def error(x_true, x_r, N):
    frobenius_norm = np.sum(np.square(x_true - x_r))
    return frobenius_norm / N


def plot_clusters(X, r, k, M, K):
    if k > 3:
        colors = mcolors.keys()  # get all colors
    else:
        colors = ['r', 'g', 'b']  # get the 3 basic colors
    for k in range(K):  # for each category
        plt.plot(X[r[:, k] == 1, 0], X[r[:, k] == 1, 1], '.', color=colors[k], markersize=3)  # plot data
    plt.plot(M[:, 0], M[:, 1], 'b+', mew=3, ms=25)  # plot the centers
    plt.show()


def plot(X, Minit):
    plt.plot(X[:, 0], X[:, 1], 'o', color='lightgray', markersize=1)
    plt.plot(Minit[:, 0], Minit[:, 1], 'b+', mew=3, ms=25)
    plt.show()
