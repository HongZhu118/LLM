import numpy as np

"""
NDSort - 采用ENS对种群进行非支配排序，返回种群对应排序后的结果以及非支配前沿的个数

------------------------------- Reference --------------------------------
 [1] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, An efficient approach to
 nondominated sorting for evolutionary multiobjective optimization, IEEE
 Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
 [2] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, A decision variable
 clustering based evolutionary algorithm for large-scale many-objective
 optimization, IEEE Transactions on Evolutionary Computation, 2018, 22(1):
 97-112.
"""


def nd_sort(objv: np.ndarray, *args):
    """
    :param objv: 目标值矩阵
    :param con: 约束矩阵
    :param count: 需要排序的前N个非支配个体或者种群部数量占比为N的个体
    :return : 返回 [非支配排序后各个体对应的结果, 非支配前沿的个数]
    """
    objv_temp = objv.copy()
    if len(objv_temp) == 0:
        return [], 0
    N, M = objv_temp.shape
    if len(args) == 1:
        count = args[0]
    else:
        con = args[0]
        count = args[1]
        Infeasible = np.any(con > 0, axis=1)
        objv_temp[Infeasible, :] = np.tile(
            np.amax(objv_temp, axis=0), (np.sum(Infeasible), 1)
        ) + np.tile(  # noqa
            np.sum(np.where(con < 0, 0, con)[Infeasible, :], axis=1).reshape(
                np.sum(Infeasible), 1
            ),
            (1, M),
        )
    return ENS_SS(objv_temp, count)


def ENS_SS(objv, count):
    # Use efficient non-dominated sort with sequential search (ENS-SS)
    """if count is None:
        nsort, _ = objv.shape
    elif isinstance(count, float):
        assert count <= 1, "if N'type is float, N must lower than 1"
        nsort = int(np.floor(len(objv) * count))
    else:
        nsort = count
    if nsort >= len(objv):
        nsort = len(objv)"""
    nsort = count
    # 目标值重复的个体不参与非支配排序
    objv, index, ind = np.unique(
        objv, return_index=True, return_inverse=True, axis=0
    )
    count, M = objv.shape
    frontno = np.full(count, np.inf)
    maxfront = 0
    # 对全部个体进行非支配排序
    Table, _ = np.histogram(ind, bins=np.arange(0, np.max(ind) + 2))
    while np.sum(Table[frontno < np.inf]) < np.min((nsort, len(ind))):
        maxfront += 1
        for i in range(count):
            if frontno[i] == np.inf:
                dominate = False
                for j in range(i - 1, -1, -1):
                    if frontno[j] == maxfront:
                        m = 1
                        while m < M and objv[i][m] >= objv[j][m]:
                            m += 1
                        dominate = m == M
                        if dominate or M == 2:
                            break
                if not dominate:
                    frontno[i] = maxfront
    frontno = frontno[ind]
    """print(frontno)
    # 取需要的个体
    ind = np.argsort(frontno)
    frontno = np.where(frontno <= frontno[ind[nsort - 1]], frontno, np.inf)
    print(frontno)
    print(maxfront)
    maxfront = max(np.setdiff1d(frontno, np.inf))
    print(maxfront)"""
    return [frontno, maxfront]
