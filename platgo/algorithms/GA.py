from platgo.GeneticAlgorithm import GeneticAlgorithm
from platgo.utils.fitness_single import fitness_single
from platgo.utils.selections.tournament_selection import (
    tournament_selection,
)  # noqa
from platgo.operators.OperatorGA import OperatorGA

# import torch
import numpy as np


# import transformers


class GA(GeneticAlgorithm):
    type = {
        "n_obj": "single",
        "encoding": {"real", "binary", "permutation"},
        "special": {"large/none", "constrained/none"},
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="GA",
        show_bar=False,
        sim_req_cb=None,
        ext_opt_prob_cb=None,
        debug=False,
        proC=1,
        disC=20,
        proM=1,
        disM=20,
    ) -> None:

        super(GA, self).__init__(
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
        self.proC = proC
        self.disC = disC
        self.proM = proM
        self.disM = disM

    def run_algorithm(self):

        pop = self.problem.init_pop()
        self.cal_obj(pop)
        while self.not_terminal(pop):
            MatingPool = tournament_selection(
                2, self.problem.pop_size, fitness_single(pop)
            )  # noqa
            Offspring = OperatorGA(
                pop[MatingPool],
                self.problem,
                self.proC,
                self.disC,
                self.proM,
                self.disM,

            )  # noqa

            self.cal_obj(Offspring)
            pop = pop + Offspring
            rank = np.argsort(fitness_single(pop), kind="mergesort")
            pop = pop[rank[0 : self.problem.pop_size]]
            print(np.min(pop.objv))
        return pop
