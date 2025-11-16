import numpy as np                     # Импорт библиотеки NumPy для численных вычислений
import sys                             # Импорт модуля sys (может не использоваться, но оставлен для совместимости)
from numpy.linalg import norm          # Импорт функции norm для вычисления нормы вектора
np.seterr(all='warn')                  # Настройка обработки ошибок NumPy: предупреждения вместо исключений


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]                           # Извлечение первой координаты (x) из вектора X
    y = X[1]                           # Извлечение второй координаты (y) из вектора X
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2  # Вычисление значения функции Химмельблау
    return v                           # Возврат значения функции


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dfH(X):
    x = X[0]                           # Извлечение x из вектора X
    y = X[1]                           # Извлечение y из вектора X
    v = np.copy(X)                     # Создание копии X для хранения градиента (вектора частных производных)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)      # Частная производная по x
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)      # Частная производная по y
    return v                           # Возврат градиента функции Химмельблау


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fR(X):
    x = X[0]                           # Извлечение x из вектора X
    y = X[1]                           # Извлечение y из вектора X
    v = (1 - x)**2 + 100*(y - x**2)**2 # Вычисление значения функции Розенброка
    return v                           # Возврат значения функции


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dfR(X):
    x = X[0]                           # Извлечение x из вектора X
    y = X[1]                           # Извлечение y из вектора X
    v = np.copy(X)                     # Создание копии X для хранения градиента
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)  # Частная производная по x для функции Розенброка
    v[1] = 200 * (y - x**2)            # Частная производная по y для функции Розенброка
    return v                           # Возврат градиента функции Розенброка


def H(X, tol, df):
    n = len(X)                         # Определение размерности задачи (для двумерного случая n=2)
    Hx = np.zeros((n, n))              # Инициализация матрицы Гессе нулями (n x n)
    h = tol                            # Шаг для численного дифференцирования (используется tol как шаг)

    for j in range(n):                 # Цикл по столбцам (вторая переменная дифференцирования)
        x_plus = np.copy(X)            # Создание копии X для возмущения в положительном направлении
        x_minus = np.copy(X)           # Создание копии X для возмущения в отрицательном направлении

        x_plus[j] += h                 # Добавление шага h к j-й компоненте для положительного возмущения
        x_minus[j] -= h                # Вычитание шага h из j-й компоненты для отрицательного возмущения

        df_plus = df(x_plus)           # Вычисление градиента в точке с положительным возмущением
        df_minus = df(x_minus)         # Вычисление градиента в точке с отрицательным возмущением

        for i in range(n):             # Цикл по строкам (первая переменная дифференцирования)
            # Центральная разность для аппроксимации второй частной производной ∂²f/∂x_i∂x_j
            Hx[i, j] = (df_plus[i] - df_minus[i]) / (2 * h)

    return Hx                          # Возврат приближённой матрицы Гессе


def nsearch(f, df, x0, tol):
    # NSEARCH searches for minimum using Newton method
    # 	answer_ = nsearch(f, df, x0, tol)
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

    kmax = 1000                        # Максимальное количество итераций (защита от зацикливания)
    x = np.copy(x0)                    # Инициализация текущей точки начальным приближением x0
    x_prev = x + 2 * tol               # Инициализация предыдущей точки так, чтобы условие цикла выполнилось
    neval = 0                          # Счётчик количества вычислений градиента (или функции — зависит от логики)
    coords = []                        # Список для сохранения траектории точек (для визуализации)
    k = 0                              # Счётчик итераций

    # Цикл продолжается, пока изменение между итерациями больше tol и не превышено kmax
    while (norm(x - x_prev) >= tol) and (k < kmax):
        x_prev = x                     # Сохранение текущей точки как предыдущей перед обновлением
        g = df(x)                      # Вычисление градиента в текущей точке
        neval += 1                     # Увеличение счётчика вычислений градиента
        Hx = H(x, tol, df)             # Численное вычисление матрицы Гессе в текущей точке
        p = -np.linalg.solve(Hx, g)    # Решение системы Hx * p = -g для нахождения направления Ньютона
        x = x + p                      # Обновление текущей точки: x_{k+1} = x_k + p
        coords.append(x)               # Сохранение новой точки в траекторию
        k += 1                         # Увеличение счётчика итераций

    xmin = x                           # Минимизатор — последняя точка
    fmin = f(x)                        # Значение функции в минимизаторе
    neval += 1                         # Учёт последнего вычисления функции (fmin)
    answer_ = [xmin, fmin, neval, coords]  # Формирование результата
    return answer_                     # Возврат результата