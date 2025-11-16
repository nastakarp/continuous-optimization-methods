import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from newton import *
import random


def drawdf(a, b, h, ax1):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [df(i) for i in x_]

    ax1.plot(x_, y, lw=1, c=color)
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)

    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")

    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax1.set_xlim([-2.1, 7.1])
    ax1.set_ylim([-30, 40])


def drawf(a, b, h, ax2):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [f(i) for i in x_]

    ax2.plot(x_, y, lw=1, c=color)
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)

    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")

    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax2.set_xlim([-2.1, 7.1])


def drawpointsDF(coord, ax1, a, b, h, i):
    ax1.plot(coord, df(coord), marker='*')
    ax1.text(coord, df(coord) + 4, str(i + 1), backgroundcolor='white')
    # tangent
    x_ = np.arange(a, b + h, h)
    new_color = (random.random(), random.random(), random.random())
    k = ddf(coord)
    br = df(coord)
    yt = k * (x_ - coord) + br
    ax1.plot(x_, yt, c=new_color, lw=1)
    ax1.plot([coord, coord], [0, df(coord)], c=(0, 0, 0), lw=0.5)


def drawpointsF(coord, ax2, i):
    ax2.plot(coord, f(coord), marker='*')
    ax2.text(coord, f(coord) + 4, str(i + 1), backgroundcolor='white')


def newtondrawfig(interval, coord):
    # draw graphics
    fig = plt.figure(figsize=[20, 10])
    # Для удобства анализа получившихся изображений можно настроить размер и пропорции выводимого окна - параметр figsize = [x, x] или width=\"XXX px\
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Newton visualisation')

    a = interval[0]
    b = interval[1]
    h = (b - a) / 100

    # plot of df function

    drawdf(a, b, h, ax1)
    drawf(a, b, h, ax2)
    print("F(x) and F'(x):")
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"
    print(ad)

    DictName = []
    print("Steps:")
    nfigs = min([10, len(coord)])  # output 10 figures or less if number of points is less
    for i in range(nfigs):
        drawpointsDF(coord[i], ax1, a, b, h, i)
        drawpointsF(coord[i], ax2, i)

        ax1.set_title(str(i + 1) + " " + 'Iteration')
        DictName.append("iter" + str(i + 1) + ".png")
        fig.savefig(DictName[i])

    for elem in DictName:
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"
        print(ad)


