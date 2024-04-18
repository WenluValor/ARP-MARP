import numpy as np
from abc import ABC, abstractmethod


class Bandit(ABC):

    def __init__(self, m, mu):
        self.m = m  # num of arms
        self.mu = mu  # 1-D vector, arm / shape = (m)

    @abstractmethod
    def feedback(self, arm):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, m, mu, n_variables):
        super().__init__(m, mu)
        self.n_variables = n_variables

    def feedback(self, arm):
        # return reward
        if isinstance(arm, list):
            sum_y = []
            for j in arm:
                p = self.mu[int(j)]
                sum_y.append(np.random.binomial(n=self.n_variables, p=p, size=1) / self.n_variables)
        else:
            j = int(arm)
            p = self.mu[int(j)]
            sum_y = np.random.binomial(n=self.n_variables, p=p, size=1) / self.n_variables
        return sum_y


class GaussianBandit(Bandit):

    def __init__(self, m, mu, sigma):
        super().__init__(m, mu)
        self.sigma = sigma

    def feedback(self, arm):
        # return reward
        if isinstance(arm, list):
            sum_y = []
            for j in arm:
                p = self.mu[int(j)]
                tmp = np.random.normal(loc=p, scale=self.sigma, size=1)
                if tmp < 0:
                    tmp = 0
                elif tmp > 1:
                    tmp = 1
                sum_y.append(tmp)
        else:
            j = int(arm)
            p = self.mu[int(j)]
            tmp = np.random.normal(loc=p, scale=self.sigma, size=1)
            if tmp < 0:
                tmp = 0
            elif tmp > 1:
                tmp = 1
            sum_y = tmp
        return sum_y


class BetaBandit(Bandit):

    def __init__(self, m, mu):
        super().__init__(m, mu)
        self.beta = 1 / mu - 1
        self.alpha = 1

    def feedback(self, arm):
        # return reward
        if isinstance(arm, list):
            sum_y = []
            for j in arm:
                p = self.beta[int(j)]
                tmp = np.random.beta(a=self.alpha, b=p, size=1)
                sum_y.append(tmp)
        else:
            j = int(arm)
            p = self.beta[int(j)]
            tmp = np.random.beta(a=self.alpha, b=p, size=1)
            sum_y = tmp
        return sum_y
