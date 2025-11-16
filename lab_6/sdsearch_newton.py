import numpy as np
import sys
from numpy.linalg import norm


def newton_1d(f1dim, x0, tol, max_iter=100):
    x = x0
    h = 1e-8
    neval = 0

    for i in range(max_iter):
        df1 = (f1dim(x + h) - f1dim(x - h)) / (2 * h)
        neval += 2

        d2f1 = (f1dim(x + h) - 2 * f1dim(x) + f1dim(x - h)) / (h ** 2)
        neval += 3

        if abs(d2f1) < 1e-14:
            print("Warning: Second derivative is too small, stopping Newton.")
            break

        x_new = x - df1 / d2f1

        if abs(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x, f1dim(x), neval


# F_HIMMELBLAU is a Himmelblau function
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    return v


def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def sdsearch(f, df, x0, tol):
    # SDSEARCH searches for minimum using steepest descent method
    # 	answer_ = sdsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    kmax = 1000
    x = x0
    coords = []
    coords.append(x)
    neval = 0
    k = 0
    deltaX = np.inf
    while (norm(deltaX) >= tol) and (k < kmax):
        grad = df(x)
        neval += 1
        f1dim = lambda alpha: f(x - alpha * grad)
        alpha_opt, _, neval_1d = newton_1d(f1dim, x0=0.01, tol=tol)  # Initial guess for alpha
        neval += neval_1d
        x_new = x - alpha_opt * grad
        deltaX = x_new - x
        coords.append(x_new)
        x = x_new
        k += 1
    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

