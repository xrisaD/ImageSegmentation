from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames as mcolors
from matplotlib import cm


def read_image():
    # creating a image object
    im = Image.open("im.jpg")
    width, height = im.size
    im_np = np.array(im.getdata())
    im_np = im_np / 255  # normalization
    return im_np, width, height


def show_image(im):
    im = im * 255
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im.show()


def init(K, D):
    # P, M , S
    return np.random.rand(K), np.random.randn(K, D), np.random.rand(K)


def initS(K):
    return np.random.uniform(low=0.1, high=1, size=(K,))


def error(x_true, x_r, N):
    frobenius_norm = (np.sum(np.square(x_true - x_r)))
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


def plot_costs(costs):
    x = range(1, len(costs) + 1)
    y = costs
    plt.plot(x, y)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Cost Function =")
    plt.xticks(x)
    plt.show()
