import numpy as np
from utils import init, initS, show_image
import math


# shapes:
# P: K
# S: K
# X: N*D
# M: K*D
# G: N*K

class EMSlowVersion:

    def __init__(self, X, K, plot=False, test=False, P=None, M=None):
        # step 1: initialization
        self.K = K
        self.N, self.D = X.shape
        self.X = X  # N*D

        if not test:
            self.P, self.M, self.S = init(self.K, self.D)
        else:
            self.P = P
            self.M = M
            self.S = initS(self.K)
        self.G = np.zeros((self.N, self.K))  # N*K

        self.plot = plot
        self.test()
    def EM_algorithm(self):
        tol = 1e-6
        Lold = np.inf
        maxIters = 15
        Ls = []
        for it in range(maxIters):
            # step 2: E step
            self.computeG()

            # step 3: M step
            self.computeM()
            self.computeS()
            self.computeP()

            # compute log likelihood
            # convergence test
            Lnew = self.likelihood()
            Ls.append(Lnew)

            print("Iteration #{}, Likelihood: {}".format(it, Lnew))
            # go to step 2 or stop
            if abs(Lnew - Lold) < tol:
                break
            Lold = Lnew

    def likelihood(self):
        likelihood = 0
        for n in range(self.N):
            p_x = 0
            for k in range(self.K):
                res = 1
                for d in range(self.D):
                    part1 = 1 / (math.sqrt(2 * math.pi * self.S[k]**2))
                    part2 = (-(self.X[n, d] - self.M[k, d])** 2) / (2 * (self.S[k])**2)
                    exp_part2 = math.exp(part2)
                    res = res * part1 * exp_part2
                p_x = p_x + self.P[k] * res
            likelihood = likelihood + np.log(p_x)
        return likelihood

    def computeG(self):
        for n in range(self.N):
            for k in range(self.K):
                res = 1
                for d in range(self.D):
                    part1 = 1 / (math.sqrt(2 * math.pi * self.S[k]**2))
                    part2 = (-(self.X[n, d] - self.M[k, d]) ** 2) / (2 * (self.S[k]) ** 2)
                    exp_part2 = math.exp(part2)
                    res = res * part1 * exp_part2
                self.G[n, k] = self.P[k] * res
            denom = np.sum(self.G[n, :])
            self.G[n, :] = self.G[n, :] / denom

    def computeM(self):
        for k in range(self.K):
            num = 0
            denom = 0
            for n in range(self.N):
                num = num + self.G[n, k] * self.X[n, :]
                denom = denom + self.G[n, k]
            print("NUM",num)
            print("DEMON", denom)
            self.M[k, :] = num / denom
        print(self.M)

    def computeS(self):
        for k in range(self.K):
            num = 0
            denom = 0
            for n in range(self.N):
                for d in range(self.D):
                    num = num + self.G[n, k] * (self.X[n, d] - self.M[k, d]) ** 2
                denom = denom + self.G[n, k]
            self.S[k] = num / (self.D * denom)
        print(self.S)
    def computeP(self):
        for k in range(self.K):
            num = 0
            for n in range(self.N):
                num = num + self.G[n, k]
            self.P[k] = num / self.N

    def show(self, width, height):
        result = np.argmax(self.G, axis=1)
        final_im = self.M[result, :]
        final_im = final_im.reshape((width, height, final_im.shape[1]))
        show_image(final_im)
