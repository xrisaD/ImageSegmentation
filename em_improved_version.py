import numpy as np
import math
from em import EM


class EMImprovedVersion(EM):

    def computeN(self):
        sub = (np.expand_dims(self.X, axis=1) - self.M) ** 2  # N*K*D
        exp = - sub / (2 * (np.expand_dims(self.S, axis=1) ** 2))
        e = np.exp(exp)
        res = e / np.sqrt(2 * np.pi * (np.expand_dims(self.S, axis=1) ** 2))
        res = np.sum(res, axis=-1)

        return res

    def likelihood(self):
        return np.sum(np.log(np.sum(self.P * self.computeN(), axis=1)))

    def computeG(self):
        num = self.P * self.computeN()
        denom = np.expand_dims(np.sum(num, axis=1), axis=1)

        self.G = num / denom

    def computeM(self):
        G_T = np.transpose(self.G)
        num = G_T @ self.X
        denom = np.expand_dims(np.sum(G_T, axis=1), 1)

        self.M = num / denom

    def computeS(self):
        num = np.sum(self.G * np.sum((np.expand_dims(self.X, axis=1) - self.M) ** 2, axis=-1), axis=0)
        denom = self.D * (np.sum(self.G, axis=0))
        self.S = np.sqrt(num / denom)

    def computeP(self):
        num = np.sum(np.transpose(self.G), axis=-1)
        self.P = num / self.N
