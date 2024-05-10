"""
------------------------------- Reference -----------------------------
 K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, A fast and elitist
 multiobjective genetic algorithm: NSGA-II, IEEE Transactions on
 Evolutionary Computation, 2002, 6(2): 182-197.
"""

import numpy as np
from .. import utils, operators, GeneticAlgorithm


class NSGA2(GeneticAlgorithm):
    type = {
        "n_obj": "multi",
        "encoding": {
            "real",
            "binary",
            "permutation",
            "label",
            "vrp",
            "two_permutation",
        },  # noqa
        "special": "constrained/none",
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="NSGA2",
        show_bar=True,
        sim_req_cb=None,
        ext_opt_prob_cb=None,
        debug=False,
    ):
        super(NSGA2, self).__init__(
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            ext_opt_prob_cb=ext_opt_prob_cb,
            debug=debug,
        )

    def run_algorithm(self):
        pop = self.problem.init_pop()
        self.cal_obj(pop)
        _, frontno, cd = self._environmental_selection(pop)  # noqa: E501

        while self.not_terminal(pop):
            matingpool = utils.tournament_selection(
                2, pop.pop_size, frontno, -cd
            )  # noqa: E501
            offspring = operators.OperatorGA(pop[matingpool], self.problem)
            self.cal_obj(offspring)
            temp_pop = offspring + pop
            pop, frontno, cd = self._environmental_selection(temp_pop)
        return pop

    def _environmental_selection(self, pop):
        """
        The environmental selection of NSGA-II
        """
        # nd_sort
        if self.problem.n_constr > 0:
            frontno, maxfront = utils.nd_sort(
                pop.objv, pop.cv, self.problem.pop_size
            )  # noqa: E501
        else:
            frontno, maxfront = utils.nd_sort(pop.objv, self.problem.pop_size)
        next = frontno < maxfront
        # calculate the crowding distance
        cd = utils.crowding_distance(pop, frontno)

        # select the soltions in the last front based on their crowding distances  # noqa: E501
        last = np.argwhere(frontno == maxfront)
        rank = np.argsort(-cd[last], axis=0)
        next[
            last[rank[0 : (self.problem.pop_size - np.sum(next))]]  # noqa
        ] = True
        # pop for next gemetation # noqa
        pop = pop[next]
        frontno = frontno[next]
        cd = cd[next]
        return pop, frontno, cd
