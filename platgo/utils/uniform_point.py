from functools import reduce
from itertools import combinations
import numpy as np
from typing import Tuple
from scipy.special import comb

"""
 uniform_point - Generate a set of uniformly distributed points.

    W,L = uniform_point(N,M) returns approximately N uniformly distributed
    points with M objectives on the unit hyperplane via the normal-boundary
    intersection method with two layers. Note that the number of sampled
    points L may be slightly smaller than the predefined size N due to the
    need for uniformity.

    W,L = uniform_point(N,M,'ILD') returns approximately N uniformly
    distributed points with M objectives on the unit hyperplane via the
    incremental lattice design. Note that the number of sampled points L
    may be slightly larger than the predefined size N due to the need for
    uniformity.

    W = uniform_point(N,M,'MUD') returns exactly N uniformly distributed
    points with M objectives on the unit hyperplane via the mixture uniform
    design method.

    W,L = uniform_point(N,M,'grid') returns approximately N uniformly
    distributed points with M objectives in the unit hypercube via the grid
    sampling. Note that the number of sampled points L may be slighly
    larger than the predefined size N due to the need for uniformity.

    W = uniform_point(N,M,'Latin') returns exactly N randomly distributed
    points with M objectives in the unit hypercube via the Latin hypercube
    sampling method.

    Example:
       W,N = uniform_point(275,10)
       W,N = uniform_point(286,10,'ILD')
       W,N = uniform_point(102,10,'MUD')
       W,N = uniform_point(1000,3,'grid')
       W,N = uniform_point(103,10,'Latin')

 ------------------------------------ Reference -------------------------------
 [1] Y. Tian, X. Xiang, X. Zhang, R. Cheng, and Y. Jin, Sampling reference
 points on the Pareto fronts of benchmark multi-objective optimization
 problems, Proceedings of the IEEE Congress on Evolutionary Computation,
 2018.
 [2] T. Takagi, K. Takadama, and H. Sato, Incremental lattice design
 of weight vector set, Proceedings of the Genetic and Evolutionary
 Computation Conference Companion, 2020, 1486-1494.
 -------------------------------------------------------------------------------
"""


def uniform_point(N: int, M: int, method="NBI"):
    if method == "NBI":
        return NBI(N, M)
    if method == "Latin":
        return Latin(N, M)
    if method == "MUD":
        return MUD(N, M)


def NBI(N: int, M: int) -> Tuple[np.ndarray, int]:
    """
    生成N个M维的均匀分布的权重向量
    :param N: 种群大小
    :param M: 目标维数
    :return: 返回权重向量和种群大小，种群大小可能有改变
    """
    H1 = 1
    while comb(H1 + M, M - 1, exact=True) <= N:
        H1 += 1
    W = (
        np.array(list(combinations(range(1, H1 + M), M - 1)))
        - np.tile(np.arange(M - 1), (comb(H1 + M - 1, M - 1, exact=True), 1))
        - 1
    )
    W = (
        np.hstack((W, np.zeros((W.shape[0], 1)) + H1))
        - np.hstack((np.zeros((W.shape[0], 1)), W))
    ) / H1
    if H1 < M:
        H2 = 0
        while (
            comb(H1 + M - 1, M - 1, exact=True)
            + comb(H2 + M, M - 1, exact=True)
            <= N
        ):
            H2 += 1
        if H2 > 0:
            W2 = (
                np.array(list(combinations(range(1, H2 + M), M - 1)))
                - np.tile(
                    np.arange(M - 1), (comb(H2 + M - 1, M - 1, exact=True), 1)
                )
                - 1
            )
            W2 = (np.hstack((W2, np.zeros(
                (W2.shape[0], 1)))) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2  # noqa
            W = np.vstack((W, W2 / 2 + 1 / (2 * M)))
    W = np.maximum(W, 1e-6)
    N = W.shape[0]
    return W, N


def Latin(N: int, M: int) -> Tuple[np.ndarray, int]:
    W = np.argsort(np.random.random(size=(N, M)), axis=0)
    W = (np.random.random(size=(N, M)) + W - 1) / N
    return W, N


def MUD(N, M):
    """
    通过混合均匀设计方法精确返回单位超平面上具有M个目标的N个均匀分布的点
    :param N: 种群大小
    :param M: 目标维数
    :return: 返回权重向量和种群大小，种群大小不变
    """
    a = 1 / np.tile(np.arange(M - 1, 0, -1), (N, 1))
    X = GoodLatticePoint(N, M - 1) ** a
    X = np.where(X > 10 ** -6, X, 10 ** -6)
    W = np.zeros((N, M))
    W[:, 0: M - 1] = (1 - X) * np.cumprod(X, axis=1) / X
    W[:, -1] = np.prod(X, axis=1)
    return W, N


def GoodLatticePoint(N, M):
    hm = np.where(np.gcd(np.arange(1, N + 1), N) == 1)[0] + 1
    udt = np.mod(
        np.dot(
            np.arange(1, N + 1).T.reshape(N, 1), hm.reshape(1, hm.shape[0])
        ),
        N,
    )
    udt = np.where(udt == 0, N, udt)
    nCombination = int(Cni(len(hm), M))
    if nCombination < 10 ** 4:
        Combination = np.array(
            list(combinations(np.arange(1, len(hm) + 1), M))
        )
        CD2 = np.zeros((nCombination, 1))
        for i in range(nCombination):
            UT = udt[:, Combination[i, :] - 1]
            CD2[i] = CalCD2(UT)
        minIndex = np.unravel_index(np.argmin(CD2), CD2.shape)[0]
        Data = udt[:, Combination[minIndex, :]]
    else:
        CD2 = np.zeros((N, 1))
        for i in range(N):
            temp = []
            for j in np.arange(0, M):
                temp.append((np.arange(1, N + 1) * (i + 1) ** j))
            UT = np.mod(np.array(temp).T, N)
            CD2[i] = CalCD2(UT)
        minIndex = np.unravel_index(np.argmin(CD2), CD2.shape)[0]
        temp = []
        for j in np.arange(0, M):
            temp.append((np.arange(1, N + 1) * minIndex ** j))
        Data = np.mod(np.array(temp).T, N)
        Data = np.where(Data == 0, N, Data)
    Data = (Data - 1) / (N - 1)
    return Data


def CalCD2(UT):
    N, S = UT.shape
    X = (2 * UT - 1) / (2 * N)
    CS1 = np.sum(np.prod(2 + np.abs(X - 1 / 2) - (X - 1 / 2) ** 2, axis=1))
    CS2 = np.zeros((N, 1))
    for i in range(N):
        CS2[i] = np.sum(
            np.prod(
                (
                    1
                    + 1 / 2 * np.abs(np.tile(X[i, :], (N, 1)) - 1 / 2)
                    + 1 / 2 * np.abs(X - 1 / 2)
                    - 1 / 2 * np.abs(np.tile(X[i, :], (N, 1)) - X)
                ),
                axis=1,
            )
        )
    CS2 = CS2.reshape(CS2.shape[1], CS2.shape[0])
    CS2 = np.sum(CS2)
    CD2 = (13 / 12) ** S - 2 ** (1 - S) / N * CS1 + 1 / (N ** 2) * CS2
    return CD2


def Cni(n, i):
    return reduce(lambda x, y: x * y, range(n - i + 1, n + 1)) / reduce(
        lambda x, y: x * y, range(1, i + 1)
    )
