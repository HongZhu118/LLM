import numpy as np
import math
from ..Population import Population
from .LLM_Response import GAhalf_binary
import re
import ollama

"""
 OperatorGAhalf - Crossover and mutation operators of genetic algorithm.
   This function is the same to OperatorGA, while only the first half of
   the offsprings are evaluated and returned.

   See also OperatorGA
"""


def OperatorGAhalf(pop, problem, *args) -> Population:
    if len(args) > 0:
        proC = args[0]
        disC = args[1]
        proM = args[2]
        disM = args[3]
    else:
        proC = 1
        disC = 20
        proM = 1
        disM = 20
    if isinstance(pop, Population):
        calobj = True
        pop = pop.decs
    else:
        calobj = False
    pop1 = pop[0: math.floor(pop.shape[0] / 2), :]  # noqa
    pop2 = pop[
        math.floor(pop.shape[0] / 2): math.floor(pop.shape[0] / 2)  # noqa
        * 2,  # noqa
        :,  # noqa
    ]  # noqa
    N = pop1.shape[0]  # noqa
    D = pop1.shape[1]
    if problem.encoding == "binary":
        # Genetic operators for binary encoding
        # Uniform crossover
        k = np.random.random((N, D)) < 0.5
        k[np.tile(np.random.random((N, 1)) > proC, (1, D))] = False
        Offspring = pop1.copy()
        Offspring[k] = pop2[k]
        # Bit-flip mutation
        Site = np.random.random((N, D)) < proM / D
        Offspring[Site] = ~Offspring[Site].astype(bool)

    elif problem.encoding == "label":
        # Genetic operators for label encoding
        # Uniform crossover
        k = np.random.random((N, D)) < 0.5
        k[np.tile(np.random.random((N, 1)) > proC, (1, D))] = False
        Offspring1 = pop1.copy()
        # Offspring2 = pop2.copy()
        Offspring1[k] = pop2[k]
        # Offspring2[k] = pop1[k]
        # TempOffspring = np.vstack((Offspring1, Offspring2))
        TempOffspring = Offspring1.copy()
        # Bit-flip mutation
        # Site = np.random.random((2 * N, D)) < proM / D
        Site = np.random.random((N, D)) < proM / D
        # Rand = np.random.randint(0, high=D, size=(2 * N, D))
        Rand = np.random.randint(0, high=D, size=(N, D))
        TempOffspring[Site] = Rand[Site]

        #          Repair
        # for i in range(2 * N):
        for i in range(N):
            Off = np.zeros((1, D)).astype(int)
            while not (Off > 0).all():
                fOff = (np.ones((Off.shape)) - Off).astype(int)
                if np.size(np.where(fOff > 0)[1]) > 0:
                    x = np.where(fOff > 0)[1][0]
                    Off[0, TempOffspring[i, :] == TempOffspring[i, x]] = (
                        np.max(Off) + 1
                    )
                else:
                    continue
            TempOffspring[i, :] = Off
        # Offspring = TempOffspring[0:N, :]
        Offspring = np.vstack((TempOffspring, pop2))

    elif problem.encoding == "permutation":
        # Genetic operators for permutation based encoding
        # Order crossover
        Offspring = pop1
        k = np.random.randint(0, high=D, size=N)
        for i in range(N):
            Offspring[i, k[i] + 1:] = np.setdiff1d(  # noqa
                pop2[i, :], pop1[i, : k[i] + 1], True
            )  # noqa
        k = np.random.randint(0, high=D, size=N) + 1
        s = np.random.randint(0, high=D, size=N) + 1
        for i in range(N):
            if s[i] < k[i]:
                Offspring[i, :] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : s[i] - 1],
                                        Offspring[i, k[i] - 1],
                                    )
                                ),
                                Offspring[i, s[i] - 1: k[i] - 1],  # noqa
                            )
                        ),  # noqa
                        Offspring[i, k[i]:],  # noqa
                    )
                )
            elif s[i] > k[i]:
                Offspring[i, :] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : k[i] - 1],
                                        Offspring[i, k[i]: s[i] - 1],  # noqa
                                    )
                                ),
                                Offspring[i, k[i] - 1],
                            )
                        ),  # noqa
                        Offspring[i, s[i] - 1:],  # noqa
                    )
                )
    elif problem.encoding == "vrp":
        CUSNUM = max(pop[0])
        Offspring = pop1
        Offspring_hat = Offspring[:, 1:-1]
        # 2-opt
        for i in range(N):
            count = 1
            # vrp编码转为序列编码,便于进行交叉
            for j in range(len(Offspring_hat[i])):
                if Offspring_hat[i][j] == 0:
                    Offspring_hat[i][j] = CUSNUM + count
                    count = count + 1
            # 2-opt
            k11 = np.random.randint(0, D - 2)
            k12 = np.random.randint(0, D - 2)
            start = min(k11, k12)
            end = max(k11, k12)
            if start != end:
                Offspring_hat[i][start: end + 1] = np.flipud(  # noqa
                    Offspring_hat[i][start: end + 1]  # noqa
                )
            # swap()
            k21 = np.random.randint(0, D - 2)
            k22 = np.random.randint(0, D - 2)
            if k21 != k22:
                Offspring_hat[i][k21], Offspring_hat[i][k22] = (
                    Offspring_hat[i][k22],
                    Offspring_hat[i][k21],
                )
            # 序列编码解码回vrp编码
            for j in range(len(Offspring_hat[i])):
                if Offspring_hat[i][j] > CUSNUM:
                    Offspring_hat[i][j] = 0
            Offspring[i] = np.hstack((0, np.hstack((Offspring_hat[i], 0))))
        Offspring = np.vstack((Offspring, pop2))

    elif problem.encoding == "two_permutation":
        Offspring = np.vstack((pop1, pop2))
        Offspring = pop1
        pop1_1 = pop1[:, : int(D / 2)]  # 前段
        pop1_2 = pop1[:, int(D / 2):]  # 后段# noqa
        pop2_1 = pop2[:, : int(D / 2)]  # 前段
        pop2_2 = pop2[:, int(D / 2):]  # 后段# noqa
        k = np.random.randint(0, high=int(D / 2), size=2 * N)
        k1 = np.random.randint(0, high=int(D / 2), size=2 * N)
        for i in range(N):
            Offspring[i, k[i] + 1: int(D / 2)] = np.setdiff1d(  # noqa
                pop2_1[i, :], pop1_1[i, : k[i] + 1], True
            )  # noqa
            Offspring[i, k1[i] + 1 + int(D / 2):] = np.setdiff1d(  # noqa
                pop2_2[i, :], pop1_2[i, : k1[i] + 1], True
            )  # noqa
        # Slight mutation
        k = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        k1 = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        s = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        s1 = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        for i in range(N):
            # 前段
            if s[i] < k[i]:
                Offspring[i, : int(D / 2)] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : s[i] - 1],
                                        Offspring[i, k[i] - 1],
                                    )
                                ),
                                Offspring[i, s[i] - 1: k[i] - 1],  # noqa
                            )
                        ),  # noqa
                        Offspring[i, k[i]: int(D / 2)],  # noqa
                    )
                )
            elif s[i] > k[i]:
                Offspring[i, : int(D / 2)] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : k[i] - 1],
                                        Offspring[i, k[i]: s[i] - 1],  # noqa
                                    )
                                ),
                                Offspring[i, k[i] - 1],
                            )
                        ),  # noqa
                        Offspring[i, s[i] - 1: int(D / 2)],  # noqa
                    )
                )
            # 后段
            if s1[i] < k1[i]:
                Offspring[i, int(D / 2):] = np.hstack(  # noqa
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[
                                            i,
                                            int(D / 2): s1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                        Offspring[i, k1[i] - 1 + int(D / 2)],
                                    )
                                ),
                                Offspring[
                                    i,
                                    s1[i]
                                    - 1
                                    + int(D / 2): k1[i]  # noqa
                                    - 1
                                    + int(D / 2),
                                ],
                            )
                        ),  # noqa
                        Offspring[i, k1[i] + int(D / 2):],  # noqa
                    )
                )
            elif s1[i] > k1[i]:
                Offspring[i, int(D / 2):] = np.hstack(  # noqa
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[
                                            i,
                                            int(D / 2): k1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                        Offspring[
                                            i,
                                            k1[i]
                                            + int(D / 2): s1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                    )
                                ),
                                Offspring[i, k1[i] - 1 + int(D / 2)],
                            )
                        ),  # noqa
                        Offspring[i, s1[i] - 1 + int(D / 2):],  # noqa
                    )
                )
    else:
        Offspring = np.empty((N,D))
        for i in range(N):
            off =GAhalf_binary(D,pop1[i],pop2[i])
            Offspring[i] = off



        # beta = np.zeros((N, D))
        # mu = np.random.random((N, D))
        # beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        # beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        # beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        # beta[np.random.random((N, D)) < 0.5] = 1
        # beta[np.tile(np.random.random((N, 1)) > proC, (1, D))] = 1
        # Offspring = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2))
        # Lower = np.tile(problem.lb, (N, 1))
        # Upper = np.tile(problem.ub, (N, 1))
        # Site = np.random.random((N, D)) < proM / D
        # mu = np.random.random((N, D))
        # temp = np.logical_and(Site, mu <= 0.5)
        # Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        #     (
        #         2 * mu[temp]
        #         + (1 - 2 * mu[temp])
        #         * (  # noqa
        #             1
        #             - (Offspring[temp] - Lower[temp])
        #             / (Upper[temp] - Lower[temp])  # noqa
        #         )
        #         ** (disM + 1)
        #     )
        #     ** (1 / (disM + 1))
        #     - 1
        # )  # noqa
        # temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
        # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        #     1
        #     - (
        #         2 * (1 - mu[temp])
        #         + 2
        #         * (mu[temp] - 0.5)
        #         * (  # noqa
        #             1
        #             - (Upper[temp] - Offspring[temp])
        #             / (Upper[temp] - Lower[temp])  # noqa
        #         )
        #         ** (disM + 1)
        #     )
        #     ** (1 / (disM + 1))
        # )  # noqa

    if calobj:  # noqa: E510
        Offspring = Population(decs=Offspring)
    return Offspring  # noqa
