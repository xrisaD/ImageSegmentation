from utils import initS, init
import numpy as np


# shapes:
# P: K
# S: K
# X: N*D
# M: K*D
# G: N*K

class EM:
    def __init__(self, X, K, maxIters=15, prints=False, test=False, P=None, M=None):
        # step 1: initialization
        self.K = K
        self.N, self.D = X.shape
        self.X = X  # N*D

        self.maxIters = maxIters

        if not test:
            self.P, self.M, self.S = init(self.K, self.D)
        else:
            self.P = P
            self.M = M
            self.S = initS(self.K)
        self.G = np.zeros((self.N, self.K))

        self.prints = prints

    def EM_algorithm(self):
        tol = 1e-6
        Lold = -np.inf
        Ls = []
        for it in range(self.maxIters):
            # step 2: E step
            self.computeG()

            # step 3: M step
            self.computeM()
            self.computeS()
            self.computeP()

            # compute log likelihood
            Lnew = self.likelihood()
            Ls.append(Lnew)

            if self.prints:
                print("Iteration #{}, Likelihood: {}".format(it, Lnew))
            # convergence test
            # go to step 2 or stop
            if abs(Lnew - Lold) < tol:
                break
            Lold = Lnew
        return Ls

