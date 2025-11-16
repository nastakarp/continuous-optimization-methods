import numpy as np
import sys
from numpy.linalg import norm
# Функция одномерной минимизации методом золотого сечения. f — минимизируемая одномерная функция,
 # interval — начальный интервал [a, b], tol — требуемая точность по ширине интервала.
def goldensectionsearch(f, interval, tol):
    a, b = interval    # Распаковываем границы интервала: a = левая, b = правая.
    phi = (1 + np.sqrt(5)) / 2    # Константа золотого сечения φ ≈ 1.618.
    resphi = 2 - phi    # Вспомогательная константа = 1/φ² ≈ 0.382, используется для вычисления внутренних точек.
    neval = 0    # Счётчик количества вычислений функции f.

    x1 = a + resphi * (b - a)    # Первая внутренняя точка интервала, ближе к a.
    x2 = b - resphi * (b - a)    # Вторая внутренняя точка интервала, симметричная x1 относительно центра.
    f1 = f(x1)    # Значение функции в точке x1.
    f2 = f(x2)    # Значение функции в точке x2.
    neval += 2    # Увеличиваем счётчик на 2, так как вычислили f дважды.

    while abs(b - a) > tol: # Пока ширина интервала больше заданной точности — продолжаем.
        if f1 < f2:            # Если f(x1) < f(x2), минимум находится в левой части [a, x2].
            b = x2            # Сужаем интервал: новая правая граница — x2.
            x2 = x1            # Точка x1 становится новой x2.
            f2 = f1            # Значение f(x1) переходит в f2.
            x1 = a + resphi * (b - a)            # Вычисляем новую x1 в обновлённом интервале.
            f1 = f(x1)            # Вычисляем f в новой x1.
        else:            # Иначе минимум в правой части [x1, b].
            a = x1            # Новая левая граница — x1.
            x1 = x2            # x2 становится новой x1.
            f1 = f2            # f2 переходит в f1.
            x2 = b - resphi * (b - a)            # Вычисляем новую x2.
            f2 = f(x2)            # Вычисляем f в новой x2.
        neval += 1        # За каждую итерацию цикла делается одно новое вычисление f.

    xmin = (a + b) / 2    # Приближение к минимуму — середина финального интервала.
    fmin = f(xmin)    # Значение функции в найденной точке минимума.
    answer_ = [xmin, fmin, neval]    # Формируем выход: [точка минимума, значение функции, число вычислений].
    return answer_    # Возвращаем результат.

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fH(X):    # Целевая функция Химмельблау.
    x = X[0]
    y = X[1]    # Извлекаем компоненты вектора X.
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2    # Формула функции Химмельблау.
    return v    # Возвращаем значение функции.


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):    # Градиент функции Химмельблау.
    x = X[0]
    y = X[1]
    v = np.copy(X)    # Создаём копию X для хранения градиента (сохраняем форму).
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)    # Частная производная по x.
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)    # Частная производная по y.

    return v    # Возвращаем градиент как вектор.


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):    # Целевая функция Розенброка.
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2    # Классическая форма функции Розенброка.
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):    # Градиент функции Розенброка.
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)    # ∂f/∂x = -2(1 - x) - 400x(y - x²)
    v[1] = 200 * (y - x ** 2)    # ∂f/∂y = 200(y - x²)
    return v


def sdsearch(f, df, x0, tol):
    # SDSEARCH searches for minimum using steepest descent method
    # 	answer_ = sdsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    kmax = 1000    # Максимальное число итераций (защита от зацикливания).
    x = x0    # Текущая точка инициализируется начальным значением.
    coords = []    # Список для хранения траектории (всех точек x во время оптимизации).
    coords.append(x)    # Сохраняем начальную точку в траекторию.
    neval = 0    # Счётчик общего числа вычислений градиента и функции.
    k = 0    # Счётчик итераций.
    deltaX = np.inf    # Инициализация разности между x_new и x (для условия останова).На первой итерации norm(deltaX) = inf >= tol → цикл запустится.

    while (norm(deltaX) >= tol) and (k < kmax):        # Условие продолжения: изменение x ещё велико И не превышено kmax.
        grad = df(x)        # Вычисляем градиент в текущей точке.
        neval += 1        # Увеличиваем счётчик (одно вычисление градиента).

        f1dim = lambda alpha: f(x - alpha * grad)        # Одномерная функция φ(α) = f(x + α·d) для поиска оптимального шага.

        interval = [0.0, 1.0]        # Фиксированный интервал поиска шага α ∈ [0, 1].
        alpha_opt, _, neval_1d = goldensectionsearch(f1dim, interval, tol)        # Находим оптимальный α методом золотого сечения.
        neval += neval_1d        # Добавляем число вычислений f внутри одномерного поиска.

        x_new = x - alpha_opt * grad        # Делаем шаг в новую точку.
        deltaX = x_new - x        # Вычисляем изменение позиции.

        coords.append(x_new)        # Сохраняем новую точку в траекторию.
        x = x_new        # Обновляем текущую точку.
        k += 1        # Увеличиваем счётчик итераций.

    xmin = x    # Найденная точка минимума.
    fmin = f(xmin)    # Значение функции в точке минимума.

    answer_ = [xmin, fmin, neval, coords]    # Формируем выходной список.
    return answer_    # Возвращаем результат.