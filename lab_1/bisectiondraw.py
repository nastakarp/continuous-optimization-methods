import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bisectionsearch import *


def bisectionsearch2slides(interval, coords):
    # make plot
    t1 = np.arange(interval[0], interval[1], 0.001)
    fig, ax = plt.subplots()
    plt.plot(t1, f(t1))
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig('plot.png')
    print("<img width=\"550px\" src=\"/resources/plot.png\">")

    nfigs = min([10, len(coords)])  # output 10 figures or less if number of points is less
    for i in range(nfigs):
        plt.plot(coords[i], f(coords[i]), 'ro')
        name = "plot" + str(i) + ".png"
        fig.savefig(name)
        ad = "<img width=\"550px\" src=\"/resources/" + name + "\">"
        print(ad)

    plt.plot(coords[-1], f(coords[-1]), 'bo')  # print last point blue
    fig.savefig('plotFin.png')
    print("Final:")
    print("<img width=\"550px\" src=\"/resources/plotFin.png\">")

