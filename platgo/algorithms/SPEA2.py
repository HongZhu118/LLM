"""------------------------------- Reference --------------------------------
 E. Zitzler, M. Laumanns, and L. Thiele, SPEA2: Improving the strength
 Pareto evolutionary algorithm, Proceedings of the Conference on
 Evolutionary Methods for Design, Optimization and Control with
 Applications to Industrial Problems, 2001, 95-100.
 """

import numpy as np

from scipy.spatial.distance import cdist
from .. import GeneticAlgorithm, utils, operators, Population


class SPEA2(GeneticAlgorithm):  # noqa
    type = {
        "n_obj": "multi",
        "encoding": {"real", "binary", "permutation", "label", "vrp", "two_permutation"}, # noqa
        "special": "",
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="SPEA2",
        show_bar=False,
        sim_req_cb=None,
        debug=False,
    ):
        super(SPEA2, self).__init__(
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            debug=debug,
        )

    def run_algorithm(self):
        pop = self.problem.init_pop()
        self.cal_obj(pop)
        fitness = _cal_fitness(pop)

        while self.not_terminal(pop):
            matingpool = utils.tournament_selection(2, pop.pop_size, fitness)
            p1, p2 = utils.random_selection(pop[matingpool])
            offspring = operators.OperatorGA(pop[matingpool], self.problem)
            self.cal_obj(offspring)  # 计算子代种群目标函数值
            temp_pop = offspring + pop  # 合并种群
            pop, fitness = _environmental_selection(
                temp_pop, self.problem.pop_size
            )
        return pop


def _cal_fitness(pop: Population):
    N, M = pop.decs.shape
    # 记录支配关系
    dominate = np.zeros((N, N), dtype=bool)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            k = int(np.any(pop.objv[i] < pop.objv[j])) - int(
                np.any(pop.objv[i] > pop.objv[j])
            )
            if k == 1:
                dominate[i][j] = True
            elif k == -1:
                dominate[j][i] = True

    # 计算S(i)
    s = np.sum(dominate, axis=1)

    # 计算R(i)
    r = np.zeros(N)
    for i in range(N):
        r[i] = np.sum(s[dominate[:, i]])

    # 计算D(i)
    distance = cdist(pop.objv, pop.objv)
    distance[np.eye(len(distance), dtype=bool)] = np.inf
    distance = np.sort(distance, axis=1)
    d = 1 / (distance[:, int(np.floor(np.sqrt(N)) - 1)] + 2)
    # 计算fitness
    fitness = r + d

    return fitness


def truncation(objv, K):
    # 截断策略
    distance = cdist(objv, objv)
    distance[np.eye(len(distance), dtype=bool)] = np.inf
    delete = np.zeros(len(objv), dtype=bool)

    while np.sum(delete) < K:
        remain = np.argwhere(~delete).flatten()
        temp = distance[remain]
        temp = temp[:, remain]
        temp = np.sort(temp, axis=1)
        _, rank = np.unique(temp, return_index=True, axis=0)
        delete[remain[rank[0]]] = True

    return delete


def _environmental_selection(pop, N: int):
    fitness = _cal_fitness(pop)
    next = fitness < 1
    if np.sum(next) < N:
        rank = np.argsort(fitness)
        next[rank[:N]] = True
    elif np.sum(next) > N:
        delete = truncation(pop.objv[next], np.sum(next) - N)
        temp = np.argwhere(next)
        next[temp[delete]] = False

    pop = pop[next]
    fitness = fitness[next]

    return pop, fitness
