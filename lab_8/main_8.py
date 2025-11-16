from BBsearch import *
import numpy as np
from BBDraw import *



def main():
    print("Rosenbrock function:")
    x0 = np.array([[2], [-1]])
    tol = 1e-9
    [xmin, f, neval, coords] = bbsearch(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords,  len(coords), fR)
    #contourPlot()
    #bbdDraw(coords, neval)
    # Уточнить разницу в результатах - matlab - xmin 1.00000
    #                                                1.00000
    #                                           fmin 2.8911e-20
    #                                           neval 90
    # и Питон- [[1.]
    #           [1.]] [2.93450702e-20] 90
    # вероятно связано с разницей хранения знаков и округлении
    # В таком случае тесты делать на xmin или fmin?

if __name__ == '__main__':
    main()
