from gradientsearch import grsearch
import numpy as np
from gradientDraw import *


def main():
    x0 = np.array([0, 1])
    tol = 1e-3
    [xmin, f, neval, coords] = grsearch(x0, tol)
    print(xmin, f, neval)
    draw(coords, len(coords))


if __name__ == '__main__':
    main()
