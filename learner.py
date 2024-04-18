import math
import numbers
import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    @abstractmethod
    def record(self, recom, reward):
        # update X[t], I[t], y[t]
        raise NotImplementedError

    @abstractmethod
    def update(self, phase, arm):
        # update recommendation vector
        raise NotImplementedError

    @abstractmethod
    def get_recommend(self, phase):
        # get recommended arm I[t]
        raise NotImplementedError

    @abstractmethod
    def refresh(self):
        # clear the algorithm, making it back to initialized
        raise NotImplementedError


class ARP(Learner):

    def __init__(self, m, c_star, k, lmd, T, tau):
        self.t = 0  # index
        self.m = m  # num of arms
        self.c_star = c_star
        self.k = k
        self.lmd = lmd
        self.T = T
        self.q = k
        self.theta = self.get_theta(tau=tau)  # theta_tau
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.explore_arm = list(np.arange(self.m))  # set B
        self.vec_p = np.zeros([m])
        self.counter = np.zeros([m])  # tau
        self.Loss = np.zeros([m])  # L
        self.est_reward = np.zeros([m])  # mu_hat
        self.sum_reward = np.zeros([m])
        self.type = 'ARP'

    def get_theta(self, tau):
        return 4 * self.m ** 2 / (tau * (1 - self.c_star - tau))

    def update(self, is_sample, arm=0):
        # update vector_p / explore_arm
        if is_sample:
            a = np.argmax(self.est_reward[0: arm])
            self.vec_p *= 0
            self.vec_p[arm] = self.get_p(arm=arm)
            self.vec_p[a] = 1 - self.vec_p[arm]
        else:
            q = self.q
            cur_max = max(self.est_reward)
            bound = max(cur_max, self.c_star)
            gap = math.sqrt(math.log(self.T * self.theta) / (2 * q))
            tmp_list = []
            for i in range(len(self.explore_arm)):
                if self.est_reward[int(self.explore_arm[i])] + gap >= bound:
                    tmp_list.append(int(self.explore_arm[i]))
            if len(tmp_list) == 0:
                tmp_list.append(np.argmax(self.est_reward))
            self.explore_arm = tmp_list
            self.q += 1

    def get_p(self, arm):
        M_hat = self.get_M_hat(arm=arm)
        return self.lmd / (2 * (self.c_star - M_hat) + self.lmd) * (M_hat < self.c_star) + (M_hat >= self.c_star)

    def get_M_hat(self, arm):
        num = int(sum(self.Loss[0: arm]))
        tmp = self.X[0: num] * self.y[0: num]
        return sum(tmp) / num

    def get_recommend(self, is_sample):
        # return recommended item (regardless of incentive)
        if is_sample:
            item = np.random.choice(a=np.arange(self.m), size=1, p=self.vec_p)
            return item[0]
        else:
            return self.explore_arm

    def record(self, recom, reward):
        if isinstance(recom, numbers.Number):
            t = int(self.t)
            self.X[t] = reward
            self.y[t] = 1
            self.I[t] = recom

            if recom != 0:
                self.sum_reward[recom] += reward
                self.counter[recom] += 1
            self.t += 1
            self.Loss[recom] += 1
        else:
            length = len(recom)
            t = int(self.t)
            length = min(self.X.shape[0] - t, length)

            self.X[t: t + length] = reward[0: length]
            self.y[t: t + length] = 1
            self.I[t: t + length] = recom[0: length]

            for i in range(length):
                j = int(recom[i])
                self.est_reward[j] = (self.est_reward[j] * self.q + reward[i]) / (self.q + 1)
                self.t += 1
        return

    def update_est_reward(self):
        # only in sample stage
        self.est_reward = self.sum_reward / self.k

    def refresh(self):
        self.t = 0  # index
        self.q = self.k

        T = int(self.T)
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action

        m = int(self.m)
        self.explore_arm = list(np.arange(self.m))  # set B
        self.vec_p = np.zeros([m])
        self.counter = np.zeros([m])  # tau
        self.Loss = np.zeros([m])  # L
        self.est_reward = np.zeros([m])  # mu_hat
        self.sum_reward = np.zeros([m])


class MARP(Learner):

    def __init__(self, m, c_value, T):
        self.eta = math.sqrt(8 * math.log(m) / T)
        self.t = 0  # index
        self.m = m  # num of arms
        self.c = c_value
        self.T = T
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.vec_p = np.zeros([m]) + 1 / m
        self.loss = np.zeros([m])  # l
        self.cum_loss = np.zeros([m])  # L
        self.type = 'MARP'

    def has_incentive(self):
        t = int(self.t)
        hist_X = self.X[0: t]
        hist_y = self.y[0: t]
        if sum(hist_y) == 0:
            return True

        hist_reward = sum(hist_X * hist_y) / sum(hist_y)

        if hist_reward >= self.c[t]:
            return True
        else:
            return False

    def get_recommend(self, phase=0):
        item = np.random.choice(a=np.arange(self.m), size=1, p=self.vec_p)
        return item[0]

    def update(self, phase=0, arm=0):
        self.update_est_loss()
        self.cum_loss += self.loss
        denom = sum(np.exp(-self.eta * self.cum_loss))
        if denom > 1e10:
            pass
        else:
            self.vec_p = np.exp(-self.eta * self.cum_loss) / denom

    def update_est_loss(self):
        self.loss *= 0
        t = int(self.t)
        i = int(self.I[t - 1])
        self.loss[i] = -self.X[t - 1] * self.y[t - 1] / self.vec_p[i]

    def record(self, recom, reward):
        t = int(self.t)
        if self.has_incentive():
            self.I[t] = recom
            self.y[t] = 1
            self.X[t] = reward
        else:
            self.y[t] = 0
        self.t += 1
        return

    def refresh(self):
        T = int(self.T)
        m = int(self.m)

        self.t = 0  # index
        self.m = m  # num of arms
        self.T = T
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.vec_p = np.zeros([m]) + 1 / m
        self.loss = np.zeros([m])  # l
        self.cum_loss = np.zeros([m])  # L


class elimination(Learner):

    def __init__(self, m, c_value, T, tau=10, delta=0.05):
        self.t = 0  # index
        self.m = m  # num of arms
        self.c = c_value
        self.T = T
        self.tau = tau  # C in Elimination paper, C > 0
        self.delta = delta  # (0, 1)
        self.q = 1

        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.explore_arm = list(np.arange(self.m))  # set B
        self.est_reward = np.zeros([m])  # mu_hat
        self.type = 'Elimination'

    def has_incentive(self):
        t = int(self.t)
        hist_X = self.X[0: t]
        hist_y = self.y[0: t]
        hist_reward = sum(hist_X * hist_y) / sum(hist_y)

        if hist_reward >= self.c[t]:
            return True
        else:
            return False

    def update(self, is_sample=False, arm=0):
        # update explore_arm
        bound = max(self.est_reward)
        gap = 2 * math.sqrt(math.log(self.tau * self.m * self.t ** 2 / self.delta) / self.t)
        tmp_list = []
        for i in range(len(self.explore_arm)):
            if self.est_reward[int(self.explore_arm[i])] + gap >= bound:
                tmp_list.append(int(self.explore_arm[i]))
        if len(tmp_list) == 0:
            tmp_list.append(np.argmax(self.est_reward))
        self.explore_arm = tmp_list
        self.q += 1

    def get_recommend(self, is_sample=False):
        return self.explore_arm

    def record(self, recom, reward):
        t = int(self.t)
        length = len(recom)
        length = min(self.X.shape[0] - t, length)
        if self.has_incentive():
            self.X[t: t + length] = reward[0: length]
            self.y[t: t + length] = 1
            self.I[t: t + length] = recom[0: length]

            for i in range(length):
                j = int(recom[i])
                self.est_reward[j] = (self.est_reward[j] * self.q + reward[i]) / (self.q + 1)
        else:
            self.y[t: t + length] = 0

        self.t += length
        return

    def refresh(self):
        m = int(self.m)
        T = int(self.T)

        self.t = 0  # index
        self.m = m  # num of arms
        self.q = 1
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.explore_arm = list(np.arange(self.m))  # set B
        self.est_reward = np.zeros([m])  # mu_hat


class UCB(Learner):

    def __init__(self, m, c_value, T):
        self.t = 0  # index
        self.m = m  # num of arms
        self.c = c_value
        self.T = T

        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.chosen_count = np.zeros([m])
        self.est_reward = np.zeros([m])  # mu_hat
        self.type = 'UCB'

    def has_incentive(self):
        t = int(self.t)
        hist_X = self.X[0: t]
        hist_y = self.y[0: t]
        hist_reward = sum(hist_X * hist_y) / sum(hist_y)

        if hist_reward >= self.c[t]:
            return True
        else:
            return False

    def get_recommend(self, is_sample=False):
        bound = self.est_reward + np.sqrt(2 * math.log(sum(self.chosen_count)) / self.chosen_count)
        item = np.argmax(bound)
        return item

    def record(self, recom, reward):
        t = int(self.t)
        if self.has_incentive():
            self.I[t] = recom
            self.y[t] = 1
            self.X[t] = reward

            i = int(recom)
            self.chosen_count[i] += 1
            self.est_reward[i] = \
                ((self.chosen_count[i] - 1) * self.est_reward[i] + self.X[t]) / (self.chosen_count[i])
        else:
            self.y[t] = 0

        self.t += 1
        return

    def update(self, phase, arm):
        pass

    def refresh(self):
        m = int(self.m)
        T = int(self.T)
        self.t = 0  # index

        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.chosen_count = np.zeros([m])
        self.est_reward = np.zeros([m])  # mu_hat


class Thompson(Learner):

    def __init__(self, m, c_value, T):
        self.t = 0  # index
        self.m = m  # num of arms
        self.c = c_value
        self.T = T

        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.success = np.zeros([m])
        self.failure = np.zeros([m])  # mu_hat
        self.type = 'Thompson'

    def has_incentive(self):
        t = int(self.t)
        hist_X = self.X[0: t]
        hist_y = self.y[0: t]
        hist_reward = sum(hist_X * hist_y) / sum(hist_y)

        if hist_reward >= self.c[t]:
            return True
        else:
            return False

    def get_recommend(self, is_sample=False):
        samples = np.random.beta(self.success + 1, self.failure + 1)
        item = np.argmax(samples)
        return item

    def record(self, recom, reward):
        t = int(self.t)
        if self.has_incentive():
            self.I[t] = recom
            self.y[t] = 1
            self.X[t] = reward

            i = int(recom)
            self.success[i] += 1
        else:
            self.y[t] = 0

            i = int(recom)
            self.failure[i] += 1

        self.t += 1
        return

    def update(self, phase, arm):
        pass

    def refresh(self):
        m = int(self.m)
        T = int(self.T)
        self.t = 0  # index
        self.X = np.zeros([T]) - 1  # reward
        self.I = np.zeros([T]) - 1  # recommendation
        self.y = np.zeros([T]) - 1  # action
        self.success = np.zeros([m])
        self.failure = np.zeros([m])  # mu_hat
