


import numpy as np


def f(x):
    return 2 * (x ** 2) - 9 * x - 31

def df(x):
    return 4 * x - 9


def bsearch(interval, tol):
    """
    Searches for minimum using bisection method (dichotomy).
    STRICTLY follows assignment: uses df(a) for stopping criterion.

    Args:
        interval: [a, b] - search interval
        tol: tolerance for stopping

    Returns:
        [xmin, fmin, neval, coords]
    """
    a, b = interval
    coords = []
    neval = 0
    L = b - a
    g = df(a)
    neval += 1
    while (np.abs(L) > tol) and (np.abs(g) > tol):
        x = (a + b) / 2
        coords.append(x)
        g = df(x)
        neval += 1
        if g > 0:
            b = x
        else:
            a = x
        L = b - a
    xmin = (a + b) / 2
    fmin = f(xmin)
    neval += 1
    answer_ = [xmin, fmin, neval, coords]
    return answer_