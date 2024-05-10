import numpy as np
from ..Problem import Problem


class SOP_F20(Problem):
    type = {
        "n_obj": "single",
        "encoding": "real",
        "special": "expensive/none",
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "SOP_F20",
            "encoding": "real",
            "n_var": 6,
            "lower": "0",
            "upper": "1",
            "n_obj": 1,
            "initFcn": [],
            "decFcn": [],
            "conFcn": [],
            "objFcn": [],
        }
        optimization_problem.update(in_optimization_problem)
        super(SOP_F20, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        a = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        c = np.array([[1], [1.2], [3], [3.2]])
        p = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        objv = np.zeros((pop.decs.shape[0], 1))
        for i in range(pop.decs.shape[0]):
            objv[i, 0] = -np.sum(
                c
                * np.exp(
                    np.reshape(
                        -np.sum(
                            a * ((np.tile(pop.decs[i, :], (4, 1)) - p)) ** 2,
                            axis=1,
                        ),
                        (-1, 1),
                    )
                )
            )
        pop.objv = objv
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.decs.shape[0], 1))