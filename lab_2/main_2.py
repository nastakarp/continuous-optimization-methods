from goldensearch import gsearch
from drawplot import gs2slides

print("Find:")
interval = [-2, 10]
tol = 1e-10
[xmin, fmin, neval, coords] = gsearch(interval,tol)
print([xmin, fmin, neval])

gs2slides(interval,coords)
'''
print("\nИтерации метода золотого сечения:")
print(f"{'Шаг':>3} | {'a':>10} {'b':>10} | {'x1':>10} {'x2':>10} | {'L = b-a':>12}")
print("-" * 60)
for i, (x1, x2, a, b) in enumerate(coords):
    L = b - a
    print(f"{i:3d} | {a:10.6f} {b:10.6f} | {x1:10.6f} {x2:10.6f} | {L:12.6e}")
'''