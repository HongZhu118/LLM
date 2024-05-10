import numpy as np
from ..Problem import Problem


# objFcn 返回的是一维数组
class DTLZ2(Problem):
    type = {
        "n_obj": {"multi", "many"},
        "encoding": "real",
        "special": {"large/none", "expensive/none"},
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "DTLZ2",
            "encoding": "real",
            "n_obj": 3,
            "n_var": 12,
            "lower": "0",
            "upper": "1",
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],
            "conFcn": ["0"],
        }
        # optimization_problem["n_var"] = optimization_problem["n_obj"] + 9
        optimization_problem.update(in_optimization_problem)
        super(DTLZ2, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        decs = pop.decs
        XM = decs[:, (self.n_obj - 1):]
        g = np.sum(((XM - 0.5) ** 2), axis=1, keepdims=True)
        ones_matrix = np.ones((pop.decs.shape[0], 1))
        f = np.fliplr(np.cumprod(np.hstack([ones_matrix, np.cos(decs[:, :self.n_obj-1] * np.pi / 2)]), 1)) * \
            np.hstack([ones_matrix, np.sin(decs[:, range(  # noqa
                self.n_obj - 2, -1, -1)] * np.pi / 2)]) * np.tile(1 + g, (1, self.n_obj))  # noqa
        pop.objv = f
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.pop_size, self.n_constr))
