import numpy as np

class EMImprovedVersion:

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

    def likelihood(self):
        power_s = 2 * ((self.S) ** 2)
        p1 = - 1 / power_s  # K
        p2 = 1 / np.sqrt(np.pi * power_s)  # K

        sub = (np.expand_dims(self.X, axis=1) - self.M) ** 2  # N * K * D
        p1 = np.expand_dims(p1, axis=1)
        p = np.exp(sub / p1)

        n = np.expand_dims(p2, axis=1) * p
        prod = np.prod(n, axis=-1)
        mul = np.sum(prod, axis=1)
        likelihood = np.sum(np.log(mul))
        return likelihood

    def EM_algorithm(self):

        # step 2: E step

        # step 3: M step

        # compute log likelihood

        # convergence test

        # go to step 2 or stop
        return 0

# An tupikes apokliseis<timi na kaneis reset th diakimansi me megali timi