import numpy as np

def f(x):
    return x**2 - 10*np.cos(0.3*np.pi*x) - 20

def df(x):
    return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def ssearch(interval, tol):
    a, b = interval
    dfa = df(a)
    dfb = df(b)
    neval = 2
    coords = []

    x = a - dfa * (b - a) / (dfb - dfa)
    dfx = df(x)
    neval += 1
    coords.append([x, a, b])

    while abs(dfx) > tol and abs(b - a) > tol:
        if dfx > 0:
            b = x
            dfb = dfx
        else:
            a = x
            dfa = dfx
        x = a - dfa * (b - a) / (dfb - dfa)
        dfx = df(x)
        neval += 1
        coords.append([x, a, b])

    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_