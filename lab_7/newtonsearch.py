import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x)**2 + 100*(y - x**2)**2
    return v

# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v


def H(X, tol, df):
    n = len(X)
    Hx = np.zeros((n, n))
    h = tol

    for j in range(n):
        x_plus = np.copy(X)
        x_minus = np.copy(X)

        x_plus[j] += h
        x_minus[j] -= h

        df_plus = df(x_plus)
        df_minus = df(x_minus)

        for i in range(n):
            Hx[i, j] = (df_plus[i] - df_minus[i]) / (2 * h)

    return Hx



def nsearch(f, df, x0, tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(f, df, x0, tol)
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
    x = np.copy(x0)
    x_prev = x + 2 * tol
    neval = 0
    coords = []
    k = 0


    while (norm(x - x_prev) >= tol) and (k < kmax):
        x_prev = x
        g = df(x)
        neval += 1
        Hx = H(x, tol, df)
        p = -np.linalg.solve(Hx, g)
        x = x + p
        coords.append(x)
        k += 1

    xmin = x
    fmin = f(x)
    neval += 1
    answer_ = [xmin, fmin, neval,  coords]
    return answer_
