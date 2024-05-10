import numpy as np


"""
tournament_selection - Tournament selection.

    P = tournament_selection(K,N,fitness1,fitness2,...) returns the indices
    of N solutions by K-tournament selection based on their fitness values.
    In each selection, the candidate having the minimum fitness1 value will
    be selected; if multiple candidates have the same minimum value of
    fitness1, then the one with the smallest fitness2 value is selected,
    and so on.

    Example:
        P = tournament_selection(2,100,FrontNo)
"""


def tournament_selection(K: int, N: int, *args) -> np.ndarray:
    for i in range(len(args)):
        args[i].reshape(1, -1)
    fitness = np.vstack(args)
    _, rank = np.unique(fitness, return_inverse=True, axis=1)
    parents = np.random.randint(low=0, high=len(rank), size=(N, K))
    best = np.argmin(rank[parents], axis=1)
    index = parents[np.arange(N), best]
    return index
