from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames as mcolors


def read_image(path):
    im = Image.open(path)
    width, height = im.size
    mode = im.mode
    im_np = np.array(im.getdata()) / 255
    shape = (height, width, im_np.shape[1])
    return im_np, shape, mode


def create_image(G, M):
    return M[np.argmax(G, axis=1), :]


def get_image(final_im, shape, mode):
    final_im = Image.fromarray(np.uint8(final_im.reshape(shape)*255), mode)
    return final_im


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


def plot_likelihoods(costs):
    x = range(1, len(costs) + 1)
    y = costs
    plt.plot(x, y)
    plt.ylabel('likelihood')
    plt.xlabel('iterations')
    plt.title("Likelihood Function")
    plt.xticks(x)
    plt.show()
