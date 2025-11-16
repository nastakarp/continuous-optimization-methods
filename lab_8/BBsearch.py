import numpy as np
from numpy.linalg import norm

def goldensectionsearch(f, interval, tol):
    a, b = interval
    gr = (1 + np.sqrt(5)) / 2
    resphi = 2 - gr
    h = abs(b - a)
    if h <= tol:
        xm = (a + b) / 2
        return [xm, f(xm), 0]
    c = a + resphi * h
    d = b - resphi * h
    yc = f(c)
    yd = f(d)
    neval = 2
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
    xm = (a + b) / 2
    fmin = f(xm)
    neval += 1
    return [xm, fmin, neval]

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

def bbsearch(f, df, x0, tol):
    kmax = 1000
    Delta = 0.1
    x_k = np.array(x0, dtype=float).flatten()
    g_k = df(x_k).flatten()
    neval = 1
    coords = [np.copy(x_k)]
    alpha_initial_search, _, neval_ls = goldensectionsearch(lambda alpha: f(x_k - alpha * g_k), (0, 1), tol)
    neval += neval_ls
    x_k1 = x_k - alpha_initial_search * g_k
    g_k1 = df(x_k1).flatten()
    neval += 1
    coords.append(np.copy(x_k1))
    deltaX = x_k1 - x_k
    g_prev = g_k
    g_curr = g_k1
    x_prev = x_k
    x_curr = x_k1
    k = 1
    while norm(deltaX) >= tol and k < kmax:
        dx = x_curr - x_prev
        dg = g_curr - g_prev
        alpha_bb = np.dot(dx, dx) / np.dot(dx, dg)
        alpha_stab = Delta / norm(g_curr)
        alpha_k = min(alpha_bb, alpha_stab)
        x_new = x_curr - alpha_k * g_curr
        x_prev = x_curr
        g_prev = g_curr
        deltaX = x_new - x_curr
        g_new = df(x_new).flatten()
        neval += 1
        x_curr = x_new
        g_curr = g_new
        coords.append(np.copy(x_curr))
        k += 1
    xmin = x_curr
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_