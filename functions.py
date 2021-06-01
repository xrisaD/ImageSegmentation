from tqdm import tqdm

from em_improved_version import EMImprovedVersion
from utils import create_image, error, get_image, MyImage

import matplotlib.pyplot as plt


def run_EM(im, K, maxIters):
    # load image
    im = MyImage(im)
    likelihoods_final_ims = []
    errors = []
    for k in tqdm(K):
        l_im = []

        em = EMImprovedVersion(im.get_image(), k, maxIters)

        # run expectation maximization algorithm and get likelihood per iteration
        Ls = em.EM_algorithm()
        l_im.append(Ls)

        # create image and compute error
        final_im = create_image(em.G, em.M)
        err = error(im.get_image(), final_im, em.N)
        errors.append(err)
        final_im = get_image(final_im, im)
        l_im.append(final_im)

        likelihoods_final_ims.append(l_im)
    return likelihoods_final_ims, errors


def plot(K, results, errors):
    fig, axs = plt.subplots(len(K), 2, figsize=(15, 40))

    for i in range(0, len(K)):
        # plot likelihoods
        likelihoods = results[i][0]
        axs[i, 0].set_ylabel('likelihood')
        axs[i, 0].set_xlabel('iterations')
        x = range(1, len(likelihoods) + 1)
        y = likelihoods
        axs[i, 0].plot(x, y)

        # plot
        axs[i, 1].axis('off')
        axs[i, 1].imshow(results[i][1])
        # show error
        axs[i, 1].title.set_text('K: ' + str(K[i]) + ' Error: ' + str(errors[i]))
