import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goldensearch import *
import random


def drawplot(a, b, x1, x2, ax, color):
    h = (b - a) / 100
    x = np.arange(a, b + h, h)
    y = [f(i) for i in x]
    ax.plot(x, y, lw=1, c=color)
    ax.scatter([a, b], [f(a), f(b)], marker='o', c=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)

    y1 = f(x1)
    ax.plot([x1, x1], [0, y1], lw=1, c=color, marker='s', ms=4)
    y2 = f(x2)
    ax.plot([x2, x2], [0, y2], lw=1, c=color, marker='s', ms=4)
    ax.set_xlim([-2.25, 10.25])


def placelabel(x, y, deltaX, deltaY, iternumber, ax):
    ax.text(x - deltaX / 2, y + 4 * deltaY, str(iternumber), backgroundcolor='white')


def gs2slides(interval, coord):
    # data for visualisation
    a = interval[0]
    b = interval[1]
    h = (b - a) / 100
    x = np.arange(a, b + 0.2, h)
    y = [f(i) for i in x]
    miny = min(y)
    maxy = max(y)
    deltaX = (b - a) / 200
    deltaY = abs(maxy - miny) / 200
    Nfigs = 9  # параметр может быть изменён для удобства визуализации

    # plot
    fig, ax = plt.subplots(figsize=[30, 13])
    fig.suptitle('Golden section visualisation')
    placelabel(a, 0, deltaX, deltaY, 1, ax)
    placelabel(b, 0, deltaX, deltaY, 1, ax)
    color = (random.random(), random.random(), random.random())
    drawplot(a, b, coord[0][0], coord[0][1], ax, color)
    print("F(x) and F'(x):")
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"
    print(ad)

    DictName = []
    print("Steps:")

    nfigs = min([Nfigs, len(coord)])  # output Nfigs figures or less if number of points is less

    j = 0
    for i in range(nfigs):
        color = (random.random(), random.random(), random.random())

        drawplot(coord[i][2], coord[i][3], coord[i][0], coord[i][1], ax, color)

        placelabel(coord[i][0], 0, deltaX, deltaY, i + 1, ax);
        placelabel(coord[i][1], 0, deltaX, deltaY, i + 1, ax);

        ax.set_title(str(i + 1) + " " + 'Iteration')
        DictName.append("iter" + str(i + 1) + ".png")
        fig.savefig(DictName[i])
        j += 1

    ax.set_title(str(j + 1) + " " + 'Iteration')
    DictName.append("iter" + str(j + 1) + ".png")
    fig.savefig(DictName[j])

    for elem in DictName:
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"
        print(ad)