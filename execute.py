import math
import os
import numpy as np
import pandas as pd

from scipy.stats import beta
from scipy.cluster.vq import kmeans, whiten
from sklearn.utils import resample

from learner import ARP, MARP, elimination, UCB, Thompson
from bandit import BernoulliBandit, GaussianBandit, BetaBandit
from action import interact, initialize, get_regret, draw_res


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    else:
        pass


def cluster_sampled(n_samples, n_clusters, dataset, seed):
    X = dataset[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                 'f8', 'f9', 'f10', 'f11', 'treatment', 'exposure', 'visit']]
    stratify_cols = pd.concat([dataset['exposure'], dataset['visit']], axis=1)

    X_sampled = resample(
        X,
        n_samples=n_samples,
        stratify=stratify_cols,
        replace=False,
        random_state=seed
    )

    whitened = whiten(X_sampled)
    kmeaned = kmeans(whitened, n_clusters, seed=seed)
    X = X_sampled

    belongs = []
    for sample in whitened:
        distances = np.linalg.norm(sample - kmeaned[0], axis=1)
        belongs.append(np.argmin(distances))
    X.insert(X.shape[1], 'cluster', belongs)

    for i in range(n_clusters):
        tmp = X[X['cluster'] == i]
        DF = pd.DataFrame(tmp)
        DF.to_csv('real-data/' + str(i) + 'cluster.csv')
    return


def compute_mu(n_clusters, b):
    mu = np.zeros([n_clusters])
    for i in range(n_clusters):
        X = pd.read_csv('real-data/' + str(i) + 'cluster.csv', index_col=0)
        treated = X[X['exposure'] == 1]
        mu[i] = np.mean(treated['visit'])
        if math.isnan(mu[i]):
            mu[i] = 0

    DF = pd.DataFrame(mu)
    DF.to_csv('real-data/' + str(b) + 'mu.csv')
    return


def set_first_mu(b):
    # ensure mu[0] > 0.2
    mu = np.array(pd.read_csv('real-data/' + str(b) + 'mu.csv', index_col=0)).reshape(-1)
    ind = np.argsort(mu)

    for j in ind:
        if mu[j] > 0.20:
            mu[0] = mu[j]
            mu[j] = 0
            break

    DF = pd.DataFrame(mu)
    DF.to_csv('real-data/' + str(b) + 'mu.csv')


def get_theta(tau, c_star, m, bandit_setting, is_real_dt):
    if bandit_setting == 'Gauss':
        if 1 - 3 / 5 * (c_star + tau) <= 0:
            raise ValueError('Should adjust c_star, tau smaller to make theta_tau in domain.')
        return 4 * m ** 2 / (tau * (1 - 3 / 5 * (c_star + tau)))

    elif bandit_setting == 'Beta':
        if (c_star + tau) > 1 / 2:
            raise ValueError('Should adjust c_star, tau smaller to make theta_tau in domain.')
        elif (c_star + tau) <= 1 / (1 + m):
            return 4 * m ** 2 / (tau * 1)
        else:
            for i in range(1, m):
                tmp = c_star + tau
                if (tmp <= 1 / (1 + i)) & (tmp > 1 / (2 + i)):
                    return 4 * m ** 2 / (tau * (i / m))

    elif bandit_setting == 'Bernoulli':
        if not is_real_dt:
            if (c_star + tau) >= 1:
                raise ValueError('Should adjust c_star, tau smaller to make theta_tau in domain.')
            return 4 * m ** 2 / (tau * (1 - beta.cdf(x=c_star + tau, a=0.5, b=3)))
        else:
            if (c_star + tau) >= 1:
                raise ValueError('Should adjust c_star, tau smaller to make theta_tau in domain.')
            return 4 * m ** 2 / (tau * (1 - (c_star + tau)))


def get_k(theta, lmd, T, m):
    A = 9 / (2 * lmd ** 2) * math.log(20 * m / lmd)
    B = theta ** 2 * math.log(T * theta)
    return max(A, B)

def dataset_preparation(bandit_setting, is_real_dt: bool,
                        B=200, m=10):
    # dataset preparation
    B = int(B)
    m = int(m)

    if is_real_dt:
        mkdir('real-data')  # a folder to store dataset
        n_clusters = 20
        n_samples = 100000
        if os.path.exists('real-data/criteo-uplift.csv'):
            pass
        else:
            raise IOError('Download the csv.gz from https://ailab.criteo.com/criteo-uplift-prediction-dataset/' +
                          ' and unzip the compressed file into the real-data folder,'
                          ' then rename the unzipped csv as criteo-uplift.csv')

        already_clustered = True
        for b in range(B):
            if not os.path.exists('real-data/' + str(b) + 'cluster.csv'):
                already_clustered = False
                break

        already_mu = True
        for b in range(B):
            if not os.path.exists('real-data/' + str(b) + 'mu.csv'):
                already_mu = False
                break

        if not already_clustered:
            dataset = pd.read_csv('real-data/criteo-uplift.csv')
            for b in range(B):
                cluster_sampled(n_samples=n_samples, n_clusters=n_clusters, dataset=dataset, seed=int(2 * b))

        if not already_mu:
            for b in range(B):
                compute_mu(n_clusters=n_clusters, b=b)

        for b in range(B):
            set_first_mu(b=b)

    else:
        mkdir('simu-data') # a folder to store dataset

        if bandit_setting == 'Gauss':
            for b in range(B):
                np.random.seed(2 * b)
                mu = np.random.uniform(low=0, high=0.6, size=m)
                DF = pd.DataFrame(mu)
                DF.to_csv('simu-data/' + str(b) + 'mu.csv')

        elif bandit_setting == 'Beta':
            for b in range(B):
                np.random.seed(2 * b)
                beta = np.random.choice(m, m, replace=False) + 1
                mu = 1 / (1 + beta)
                DF = pd.DataFrame(mu)
                DF.to_csv('simu-data/' + str(b) + 'mu.csv')

        elif bandit_setting == 'Bernoulli':
            for b in range(B):
                np.random.seed(2 * b)
                alpha = 0.5
                beta = 3
                mu = np.random.beta(a=alpha, b=beta, size=m)
                DF = pd.DataFrame(mu)
                DF.to_csv('simu-data/' + str(b) + 'mu.csv')

def run(bandit_setting='Gauss', learner_setting='ARP', is_real_dt=False,
        m=10, B=200, tau=0.2, lmd=0.05,
        size=10, round=5000,
        c_star=0.2, c_alpha=1.0, c_beta=2.0,
        CI_alpha=0.10, Thompson_included=False,
        showdt=True, showplt=True):
    '''
    :param bandit_setting: 'Bernoulli', 'Beta', 'Gauss' (only used for synthetic data)
    :param learner_setting: 'ARP', 'MARP'; c_star known or unknown
    :param is_real_dt: True, False; indicate Criteo Uplift dataset
    :param m: int, number of arms
    :param B: int, boostrap time
    :param tau: tau in the paper, default 0.2
    :param lmd: lambda in the paper, default 0.05
    :param size: number of sample points in the line plot, default 10
    :param round: number of agents arriving, default 5000
    :param c_star: c_star in the paper, default 0.2
    :param c_alpha: c_t ~ Beta(alpha, beta) in MARP, c_alpha = alpha
    :param c_beta: c_t ~ Beta(alpha, beta) in MARP, c_beta = alpha
    :param CI_alpha: float, 0-1, Confidence Interval significance level = 1 - CI_alpha
    :param Thompson_included: bool, include Thompson sampling if True
    :param showdt: bool, show table data if True
    :param showplt: bool; show line plots if True
    :return:
    '''

    if not bandit_setting in ['Bernoulli', 'Beta', 'Gauss']:
        raise ValueError('bandit_setting must be one of the following strings: Bernoulli, Beta, Gauss.')
    if not learner_setting in ['ARP', 'MARP']:
        raise ValueError('learner_setting must be one of the following strings: ARP, MARP.')

    B = int(B)
    m = int(m)
    tau = tau
    lmd = lmd
    round = int(round)
    size = int(size)
    T = int(round)

    c_star = c_star
    c_alpha = c_alpha
    c_beta = c_beta

    dataset_preparation(bandit_setting=bandit_setting, is_real_dt=is_real_dt,
                        B=B, m=m)

    learner_list = []
    c_value = np.zeros([T])

    if learner_setting == 'ARP':
        theta = get_theta(tau=tau, c_star=c_star, m=m,
                          bandit_setting=bandit_setting, is_real_dt=is_real_dt)
        k = get_k(theta=theta, lmd=lmd, T=T, m=m)
        k = min(10, k)
        c_value = np.zeros([T]) + c_star

        learner_list.append(ARP(m=m, c_star=c_star, k=k, lmd=lmd, T=T, tau=tau))

    elif learner_setting == 'MARP':
        c_value = np.random.beta(a=c_alpha, b=c_beta, size=T)
        learner_list.append(MARP(m=m, c_value=c_value, T=T))

    learner_list.append(elimination(m=m, c_value=c_value, T=T))
    learner_list.append(UCB(m=m, c_value=c_value, T=T))
    if Thompson_included:
        learner_list.append(Thompson(m=m, c_value=c_value, T=T))

    num_type = len(learner_list)
    res = np.zeros([B, num_type, size])

    for b in range(B):
        # tbc check random seed on the server
        if is_real_dt:
            np.random.seed(2 * b)
            mu = np.array(pd.read_csv('real-data/' + str(b) + 'mu.csv', index_col=0))
            bandit_init = BernoulliBandit(m=m, mu=mu, n_variables=100000)
        else:
            np.random.seed(10 * b)
            mu = np.array(pd.read_csv('simu-data/' + str(b) + 'mu.csv', index_col=0))
            if learner_setting == 'ARP':
                mu[0] = c_star

            if bandit_setting == 'Gauss':
                bandit_init = GaussianBandit(m=m, mu=mu, sigma=0.1)
            elif bandit_setting == 'Beta':
                bandit_init = BetaBandit(m=m, mu=mu)
            elif bandit_setting == 'Bernoulli':
                bandit_init = BernoulliBandit(m=m, mu=mu, n_variables=100000)
        print('Progress: (%d / %d)' % (b + 1, B))

        for i in range(num_type):
            initialize(bandit_init, learner_list[i], learner_type=learner_list[i].type)
            interact(bandit_init, learner_list[i], round=round, learner_type=learner_list[i].type)
            for j in range(size):
                res[b, i, j] = get_regret(bandit=bandit_init, learner=learner_list[i], end=int(round / size) * j)

            learner_list[i].refresh()

    names = []
    for lr in learner_list:
        names.append(lr.type)

    if showplt:
        draw_res(res=res, names=names, round=round)

    if showdt:
        avg = np.mean(res, axis=0)
        stdev = np.std(res, axis=0)
        low = np.quantile(res, CI_alpha / 2, axis=0)
        high = np.quantile(res, 1 - CI_alpha / 2, axis=0)

        for i in range(num_type):
            print('---------' + learner_list[i].type + '--------')
            print('Mean: %.5f' % (avg[i, size - 1]))
            print('Standard Deviation: %.5f' % (stdev[i, size - 1]))
            print('%d%% Confidence Interval: (%.5f, %.5f)'
                  % (int((1 - CI_alpha) * 100), low[i, size - 1], high[i, size - 1]))

    np.save(file='res.npy', arr=res)
    # res = np.load(file='res.npy')

if __name__ == '__main__':
    run(bandit_setting='Beta', learner_setting='MARP', is_real_dt=False,
        m=10, B=5, tau=0.2, lmd=0.05,
        size=10, round=5000,
        c_star=0.1, c_alpha=0.9, c_beta=0.9,
        CI_alpha=0.10, Thompson_included=False,
        showdt=True, showplt=True)
    exit(0)
