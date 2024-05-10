import numpy as np
from ..Population import Population


"""
fitness_single - Fitness calculation for single-objective optimization.

   Fit = fitness_single(P) calculates the fitness value of each solution in
   P for single-objective optimization, where both the objective value and
   constraint violation are considered.

   Example:
       Fitness = fitness_single(Population)

------------------------------------ Reference --------------------------------
 K. Deb, An efficient constraint handling method for genetic algorithms,
 Computer Methods in Applied Mechanics and Engineering, 2000, 186(2-4):
 311-338.
 ------------------------------------------------------------------------------
"""


def fitness_single(pop: Population):
    PopCon = np.sum(np.maximum(0, pop.cv), axis=1)
    feasible = PopCon <= 0
    fitness = feasible * pop.objv.flatten() + ~feasible * (PopCon + 1e10)
    return fitness
