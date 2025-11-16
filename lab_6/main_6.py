from steepestsearch import sdsearch
import numpy as np
from steepestDraw import *



def main():
    print("Himmelblau function:")
    x0 = np.array([[1.3], [2.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = sdsearch(fH, dfH, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)
    draw(coords, len(coords), "h", fH)


    print("Rosenbrock function:")
    x0 = np.array([[1.0], [-2.0]])
    tol = 1e-7
    [xmin, f, neval, coords] = sdsearch(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords, len(coords), "r", fR)


if __name__ == '__main__':
    main()
