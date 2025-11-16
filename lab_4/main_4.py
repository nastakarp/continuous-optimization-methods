from secantdraw import *
from secantsearch import ssearch

def main():
    print("Find:")
    interval = [-2, 5]
    tol = 1e-6
    [xmin, f, neval, coords] = ssearch(interval,tol)
    print([xmin, f, neval])
    secantsearchsecants(coords, interval)


if __name__ == '__main__':
    main()
