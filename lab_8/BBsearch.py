import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    """
    Performs a golden-section search to find the minimum of a univariate function.

    Args:
        f: The function to minimize. Should take a scalar input.
        interval: A tuple (a, b) defining the initial search interval.
        tol: The tolerance for the search range.

    Returns:
        answer_: A list [xmin, fmin, neval] where:
                 xmin: The x value at the minimum.
                 fmin: The function value at xmin.
                 neval: The number of function evaluations.
    """
    a, b = interval
    # Golden ratio constants
    gr = (1 + np.sqrt(5)) / 2
    resphi = 2 - gr  # 1 / gr / gr

    h = abs(b - a)
    if h <= tol:
        # Return midpoint if interval is already small enough
        xm = (a + b) / 2
        return [xm, f(xm), 0]

    # Calculate initial points
    c = a + resphi * h
    d = b - resphi * h
    yc = f(c)
    yd = f(d)
    neval = 2

    # Main loop
    while h > tol:
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = abs(b - a)
            c = a + resphi * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = abs(b - a)
            d = b - resphi * h
            yd = f(d)
        neval += 1

    # Return the best estimate for the minimum
    xm = (a + b) / 2
    fmin = f(xm)
    # Add final function evaluation count
    neval += 1
    return [xm, fmin, neval]


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
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
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def bbsearch(f, df, x0, tol):
    """
    Searches for minimum using stabilized BB1 method.

    Args:
        f: Objective function.
        df: Gradient of the objective function.
        x0: Start point.
        tol: Tolerance for the search.

    Returns:
        answer_: A list [xmin, fmin, neval, coords] where:
                 xmin: The point at the minimum.
                 fmin: The function value at xmin.
                 neval: The number of gradient evaluations.
                 coords: Array of coordinates during optimization.
    """
    kmax = 1000
    Delta = 0.1  # Stabilization parameter

    x_k = np.array(x0, dtype=float).flatten()  # Ensure x0 is 1D
    g_k = df(x_k).flatten()  # Ensure gradient is 1D
    neval = 1
    coords = [np.copy(x_k)]

    # Initial step: Golden-section search along -g0 direction
    def line_search_func(alpha):
        return f(x_k - alpha * g_k)

    alpha_initial_search, _, neval_ls = goldensectionsearch(line_search_func, (0, 1), tol)
    neval += neval_ls
    x_k1 = x_k - alpha_initial_search * g_k
    g_k1 = df(x_k1).flatten()  # Ensure gradient is 1D
    neval += 1
    coords.append(np.copy(x_k1))

    deltaX = x_k1 - x_k
    g_prev = g_k
    g_curr = g_k1
    x_prev = x_k
    x_curr = x_k1

    k = 1
    while norm(deltaX) >= tol and k < kmax:
        # Calculate BB1 step size
        dx = x_curr - x_prev
        dg = g_curr - g_prev
        # Use np.dot for 1D vectors (standard dot product)
        alpha_bb = np.dot(dx, dx) / np.dot(dx, dg)

        # Calculate stabilization step size
        alpha_stab = Delta / norm(g_curr)

        # Take the minimum of the two
        alpha_k = min(alpha_bb, alpha_stab)

        # Update point
        x_new = x_curr - alpha_k * g_curr

        # Store previous values for next iteration
        x_prev = x_curr
        g_prev = g_curr
        deltaX = x_new - x_curr

        # Evaluate gradient at new point
        g_new = df(x_new).flatten()  # Ensure gradient is 1D
        neval += 1

        # Update current values
        x_curr = x_new
        g_curr = g_new

        coords.append(np.copy(x_curr))

        k += 1

    xmin = x_curr
    fmin = f(xmin)
    # neval already counts gradient evaluations correctly

    answer_ = [xmin, fmin, neval, coords]
    return answer_
