import numpy as np

from ..Population import Population

"""
------------------------------- Reference --------------------------------
S. Kukkonen and K. Deb, Improved pruning of non-dominated solutions based
on crowding distance for bi-objective optimization problems, Proceedings
of the 2006 IEEE Congress on Evolutionary Computation, 2006, 1179-1186.
"""


def crowding_distance(pop: Population, frontNo: np.ndarray = None) -> np.ndarray:  # noqa: E501
    """
    :param pop: 种群
    :param frontNo: 种群非支配排序后的结果
    :return : 返回 各个个体对应的拥挤距离，边界个体拥挤距离为inf
    """
    if isinstance(pop, Population):
        objv = pop.objv
    else:
        objv = pop
    N, M = objv.shape
    if frontNo is None:
        frontNo = np.ones(N)
    # 拥挤距离初始化为0
    cd = np.zeros(N)
    fronts = np.setdiff1d(np.unique(frontNo), np.inf)
    for f in range(len(fronts)):
        front = np.argwhere(frontNo == fronts[f]).flatten()
        fmax = np.max(objv[front], axis=0).flatten()
        fmin = np.min(objv[front], axis=0).flatten()
        for i in range(M):
            rank = np.argsort(objv[front, i])
            cd[front[rank[0]]] = np.inf
            cd[front[rank[-1]]] = np.inf
            for j in range(1, len(front) - 1):
                cd[front[rank[j]]] = cd[front[rank[j]]] + (
                    objv[front[rank[j + 1]]][i] - objv[front[rank[j - 1]]][i]
                ) / (fmax[i] - fmin[i])
    return cd


if __name__ == "__main__":
    pop = Population(decs=np.random.random((7, 2)))
    a = np.array(
        [
            [1.1, 1.2, 0.9],
            [1.2, 0.9, 0.8],
            [0.5, 0.4, 0.2],
            [0.7, 1.2, 0.4],
            [0.6, 1.1, 1.5],
            [1.5, 1.2, 1.1],
            [2.0, 0.2, 0.5],
        ]
    )
    pop.objv = a
    b = crowding_distance(pop)
    print(b)
