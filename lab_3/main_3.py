from newtondraw import newtondrawfig
from newton import nsearch

def main():
    print("Find:")
    interval = [-2, 7] #for drawing
    tol = 0.01
    [xmin, f, neval, coords] = nsearch(tol, 1.3)
    print([xmin, f, neval])
    newtondrawfig(interval, coords)


if __name__ == '__main__':
    main()