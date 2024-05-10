import numpy as np
import random
from ..GeneticAlgorithm import GeneticAlgorithm
from ..operators.OperatorGAhalf import OperatorGAhalf
from ..utils.uniform_point import uniform_point
from scipy.spatial.distance import cdist


class MOEAD(GeneticAlgorithm):
    type = {
        "n_obj": {"multi", "many"},
        "encoding": {"real", "binary", "permutation", "label", "vrp", "two_permutation"},  # noqa
        "special": ""
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        simulation_request_callback,
        max_fe=10000,
        ope=1,
        name=None,
        show_bar=False,
        sim_req_cb=None,
        debug=False
    ):
        super(MOEAD, self).__init__(
            pop_size,
            options,
            optimization_problem,
            simulation_request_callback,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            debug=debug
        )
        self.ope = ope

    def run_algorithm(self):
        W, N = uniform_point(
            self.problem.pop_size, self.problem.n_obj)
        T = int(np.ceil(N/10))
        B = cdist(W, W)
        B = np.argsort(B)
        B = B[:, :T]
        pop = self.problem.init_pop(N=N)
        self.cal_obj(pop)
        Z = np.nanmin(pop.objv, axis=0)

        while self.not_terminal(pop):
            for i in range(N):
                p = B[i, random.sample(range(0, B.shape[1]), B.shape[1])]
                off = OperatorGAhalf(pop[p[:2]], self.problem)
                self.cal_obj(off)
                Z = np.fmin(Z, off.objv)
                if self.ope == 1:
                    # PBI
                    normW = np.sqrt(np.sum(W[p] ** 2, axis=1))
                    normP = np.sqrt(np.sum((pop[p].objv - Z) ** 2, axis=1))
                    normO = np.sqrt(np.sum((off.objv - Z)**2, axis=1))
                    CosineP = np.sum((pop[p].objv-Z)*W[p], axis=1)/normW/normP
                    CosineO = np.sum((off.objv-Z)*W[p], axis=1)/normW/normO

                    g_old = normP*CosineP+5*normP*np.sqrt(1-CosineP**2)
                    g_new = normO*CosineO+5*normO*np.sqrt(1-CosineO**2)
                elif self.ope == 2:
                    # Tchebycheff approch
                    g_old = np.max(np.abs(pop[p].objv - Z) * W[p], axis=1)
                    g_new = np.max(np.abs(off.objv - Z) * W[p], axis=1)
                elif self.ope == 3:
                    # Tchebycheff approch with normalization
                    Zmax = np.max(pop.objv, axis=0)
                    g_old = np.max(np.abs(pop[p].objv-Z)/(Zmax-Z)*W[p], axis=1)
                    g_new = np.max(np.abs(off.objv-Z)/(Zmax-Z)*W[p], axis=1)
                else:
                    # Modified Tchebycheff approch
                    g_old = np.max(np.abs(pop[p].objv - Z) / W[p], axis=1)
                    g_new = np.max(np.abs(off.objv - Z) / W[p], axis=1)
                pop[p[g_old >= g_new]] = off

        return pop
