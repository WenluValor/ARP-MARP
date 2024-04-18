import matplotlib.pyplot as plt
import numpy as np

def get_regret(bandit, learner, end):
    # return regret of 'bandit' and 'learner' till 'end' rounds
    I_list = learner.I[0: end]
    y_list = learner.y[0: end]
    sum1 = get_l(bandit=bandit, learner=learner, I_list=I_list, y_list=y_list)
    sum2 = np.zeros([learner.m])

    y_list = np.ones([end])
    for i in range(learner.m):
        item_list = np.zeros([end]) + i
        sum2[i] = get_l(bandit=bandit, learner=learner, I_list=item_list, y_list=y_list)
    return sum1 - min(sum2)


def get_l(bandit, learner, I_list: np.array, y_list: np.array):
    # get regrets of I, y following list I_list and y_list
    sum = 0
    mat_l = get_mat_l(bandit=bandit, learner=learner)
    for i in range(I_list.shape[0]):
        i1 = int(I_list[i])
        i2 = int(y_list[i])
        sum += mat_l[i1, i2]
    return sum


def get_mat_l(bandit, learner):
    # shape of (m, 2), of arm i's regret whether or not the agent pulls
    mat_l = np.zeros([learner.m, 2])
    for i in range(learner.m):
        for j in range(2):
            mat_l[i, j] = max(bandit.mu - bandit.mu[i] * j)
    return mat_l


def draw_res(res: np.array, names, round):
    # draw the line plot of res: shape of (B, types, size)
    # B: number of bootstrap time
    # size: number of sampling points
    # types: number of algorithm included in the graph
    types = res.shape[1]
    size = res.shape[2]

    truth = np.mean(res, axis=0)

    x = np.arange(0, size)
    x *= int(round / size)
    fig, ax = plt.subplots()
    colors = np.array([[254, 67, 101], [249, 205, 173], [131, 175, 155],
                       [36, 169, 225], [78, 29, 76]])
    markers = ['o', 'v', 'x', 'D', 's']

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    for j in range(types):
        y_points = truth[j]
        ax.plot(x, y_points, label=names[j], marker=markers[j], markersize='7', color=colors[j] / 255,
                linewidth=1, alpha=1)

    ax.legend(fontsize=18)
    plt.savefig('mean_regret.png', dpi=300)

    plt.show()
    return

def initialize(bandit, learner, learner_type='MARP'):

    if learner_type == 'ARP':
        k = int(learner.k)
        arm_list = list(np.zeros([k]))
        reward = bandit.feedback(arm=arm_list)

        learner.I[0: k] = np.zeros([k])
        learner.X[0: k] = reward
        learner.y[0 : k] = 1
        learner.t = k

        learner.sum_reward[0] = sum(reward)
        learner.Loss[0] = k

    elif learner_type == 'MARP':
        for i in range(learner.m):
            t = int(learner.t)
            recom = i
            reward = bandit.feedback(arm=recom)
            learner.y[t] = 1
            learner.I[t] = recom
            learner.X[t] = reward

            learner.t += 1

    elif learner_type == 'Elimination':
        for i in range(learner.m):
            arm = i
            reward = bandit.feedback(arm=arm)

            t = int(learner.t)
            learner.I[t] = i
            learner.X[t] = reward
            learner.y[t] = 1
            learner.t += 1

            learner.est_reward[i] = reward

    elif learner_type == 'UCB':
        for i in range(learner.m):
            t = int(learner.t)
            arm = i
            reward = bandit.feedback(arm=arm)

            learner.I[t] = i
            learner.y[t] = 1
            learner.X[t] = reward

            learner.chosen_count[i] += 1
            learner.est_reward[i] \
                = ((learner.chosen_count[i] - 1) * learner.est_reward[i] + learner.X[t]) / (learner.chosen_count[i])

            learner.t += 1

    elif learner_type == 'Thompson':
        for i in range(learner.m):
            arm = i
            reward = bandit.feedback(arm=arm)

            t = int(learner.t)
            learner.I[t] = i
            learner.X[t] = reward
            learner.y[t] = 1
            learner.t += 1

    return


def interact(bandit, learner, round, learner_type='MARP'):
    # simulate I, y, X of 'bandit' using 'learner', util 'round' agents
    if learner_type == 'ARP':
        # sampling
        for i in range(1, learner.m):
            learner.update(is_sample=True, arm=i)
            while learner.counter[i] < learner.k:
                recom = learner.get_recommend(is_sample=True)
                reward = bandit.feedback(arm=recom)
                learner.record(recom=recom, reward=reward)
                if learner.t >= learner.T:
                    return
        learner.update_est_reward()

        # exploration
        while len(learner.explore_arm) > 1:
            learner.update(is_sample=False)
            recom = learner.get_recommend(is_sample=False)
            reward = bandit.feedback(arm=recom)
            learner.record(recom=recom, reward=reward)
            if learner.t >= learner.T:
                return

        # exploitation
        length = int(round - learner.t)
        recom = np.zeros([length]) + learner.explore_arm[0]
        reward = bandit.feedback(arm=list(recom))
        learner.record(recom=recom, reward=reward)

    elif learner_type == 'MARP':
        start = learner.t
        for i in range(start, round):
            learner.update()
            recom = learner.get_recommend()
            reward = bandit.feedback(arm=recom)
            learner.record(recom=recom, reward=reward)

    elif learner_type == 'Elimination':
        # exploration
        while len(learner.explore_arm) > 1:
            learner.update(is_sample=False)
            recom = learner.get_recommend(is_sample=False)
            reward = bandit.feedback(arm=recom)
            learner.record(recom=recom, reward=reward)
            if learner.t >= learner.T:
                return

        # exploitation
        length = int(round - learner.t)
        recom = np.zeros([length]) + learner.explore_arm[0]
        reward = bandit.feedback(arm=list(recom))
        learner.record(recom=recom, reward=reward)

    elif learner_type == 'UCB':
        start = learner.t
        for i in range(start, round):
            recom = learner.get_recommend()
            reward = bandit.feedback(arm=recom)
            learner.record(recom=recom, reward=reward)

    elif learner_type == 'Thompson':
        start = learner.t
        for i in range(start, round):
            recom = learner.get_recommend()
            reward = bandit.feedback(arm=recom)
            learner.record(recom=recom, reward=reward)
    return
