import numpy as np
from ..Problem import Problem


# objFcn 返回的是一维数组
class DTLZ1(Problem):
    type = {
        "n_obj": {"multi", "many"},
        "encoding": "real",
        "special": {"large/none", "expensive/none"},
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "DTLZ1",
            "encoding": "real",
            "n_var": 7,
            "lower": "0",
            "upper": "1",
            "n_obj": 3,
            "initFcn": [],
            "decFcn": [],
            "conFcn": ["0"],
            "objFcn": [],
        }
        # optimization_problem["n_var"] = optimization_problem["n_obj"] + 4
        optimization_problem.update(in_optimization_problem)
        super(DTLZ1, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        decs = pop.decs
        XM = decs[:, (self.n_obj - 1):]
        g = 100 * (self.n_var - self.n_obj + 1 + np.sum(((XM - 0.5) **
                   2 - np.cos(20 * np.pi * (XM - 0.5))), axis=1, keepdims=True))  # noqa
        ones_matrix = np.ones((pop.decs.shape[0], 1))
        f = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_matrix, decs[:, :self.n_obj - 1]]), axis=1)) * np.hstack(  # noqa
            [ones_matrix, 1 - decs[:, range(self.n_obj - 2, -1, -1)]]) * np.tile(1 + g, (1, self.n_obj))  # noqa
        pop.objv = f
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.pop_size, self.n_constr))
