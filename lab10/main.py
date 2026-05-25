import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# ЛАБОРАТОРНА РОБОТА №10
# Методи Рунге-Кутта та Адамса
# =====================================================

# -----------------------------------------------------
# ДИФЕРЕНЦІАЛЬНЕ РІВНЯННЯ
# y' = y - x^2 + 1
#
# Початкова умова:
# y(0) = 0.5
#
# Точний розв'язок:
# y = (x + 1)^2 - 0.5 * exp(x)
# -----------------------------------------------------

# -----------------------------
# ФУНКЦІЯ f(x, y)
# -----------------------------
def f(x, y):
    return y - x**2 + 1


# -----------------------------
# ТОЧНИЙ РОЗВ'ЯЗОК
# -----------------------------
def exact_solution(x):
    return (x + 1)**2 - 0.5 * np.exp(x)


# =====================================================
# ЧАСТИНА 1
# МЕТОД РУНГЕ-КУТТА 4 ПОРЯДКУ
# =====================================================

def runge_kutta_4(f, x0, y0, h, xn):

    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < xn:

        k1 = h * f(x, y)

        k2 = h * f(x + h/2, y + k1/2)

        k3 = h * f(x + h/2, y + k2/2)

        k4 = h * f(x + h, y + k3)

        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6

        x = x + h

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)


# =====================================================
# ПОХИБКА ВІДНОСНО ТОЧНОГО РОЗВ'ЯЗКУ
# =====================================================

def local_error_exact(x, y_num):

    y_exact = exact_solution(x)

    error = np.abs(y_exact - y_num)

    return error


# =====================================================
# ПОХИБКА ЗА МЕТОДОМ РУНГЕ
# =====================================================

def runge_error(f, x0, y0, h, xn):

    # крок h
    x1, y1 = runge_kutta_4(f, x0, y0, h, xn)

    # крок h/2
    x2, y2 = runge_kutta_4(f, x0, y0, h/2, xn)

    # беремо кожне друге значення
    y2_half = y2[::2]

    # для RK4:
    # error = |y(h/2) - y(h)| / (2^4 - 1)
    error = np.abs(y2_half - y1) / 15

    return x1, error


# =====================================================
# АВТОМАТИЧНИЙ ВИБІР КРОКУ
# =====================================================

def adaptive_rk4(f, x0, y0, xn, eps):

    x_values = [x0]
    y_values = [y0]
    h_values = []

    x = x0
    y = y0

    h = 0.1

    while x < xn:

        # 1 крок h
        _, y_big = runge_kutta_4(f, x, y, h, x + h)

        # 2 кроки h/2
        _, y_small = runge_kutta_4(f, x, y, h/2, x + h)

        y1 = y_big[-1]
        y2 = y_small[-1]

        # оцінка похибки
        error = abs(y2 - y1) / 15

        if error > eps:
            h = h / 2
            continue

        x = x + h
        y = y2

        x_values.append(x)
        y_values.append(y)
        h_values.append(h)

        if error < eps / 32:
            h = h * 2

    return np.array(x_values), np.array(y_values), np.array(h_values)


# =====================================================
# ЧАСТИНА 2
# МЕТОД АДАМСА
# =====================================================

def adams_predictor_corrector(f, x0, y0, h, xn):

    # Перші точки знаходимо методом RK4
    x_rk, y_rk = runge_kutta_4(f, x0, y0, h, x0 + 2*h)

    x_values = list(x_rk)
    y_values = list(y_rk)

    x = x_values[-1]

    while x < xn - h:

        n = len(x_values) - 1

        # ----------------------------------
        # ПРОГНОЗ
        # y(n+1)
        # ----------------------------------

        y_pred = y_values[n] + h * (
            3*f(x_values[n], y_values[n])
            - f(x_values[n-1], y_values[n-1])
        ) / 2

        x_next = x_values[n] + h

        # ----------------------------------
        # КОРЕКЦІЯ
        # ----------------------------------

        y_corr = y_values[n] + h * (
            f(x_next, y_pred)
            + f(x_values[n], y_values[n])
        ) / 2

        x_values.append(x_next)
        y_values.append(y_corr)

        x = x_next

    return np.array(x_values), np.array(y_values)


# =====================================================
# ГОЛОВНА ПРОГРАМА
# =====================================================

x0 = 0
y0 = 0.5

xn = 2

h = 0.1

eps = 1e-5


# =====================================================
# RK4
# =====================================================

x_rk, y_rk = runge_kutta_4(f, x0, y0, h, xn)

error_exact_rk = local_error_exact(x_rk, y_rk)

x_err, error_runge = runge_error(f, x0, y0, h, xn)


# =====================================================
# АДАМС
# =====================================================

x_adams, y_adams = adams_predictor_corrector(
    f,
    x0,
    y0,
    h,
    xn
)

error_adams = local_error_exact(x_adams, y_adams)


# =====================================================
# АВТОМАТИЧНИЙ КРОК
# =====================================================

x_auto, y_auto, h_auto = adaptive_rk4(
    f,
    x0,
    y0,
    xn,
    eps
)


# =====================================================
# ВИВІД ТАБЛИЦІ
# =====================================================

print("===== МЕТОД РУНГЕ-КУТТА =====")

print("x\t\t y_num\t\t y_exact\t error")

for i in range(len(x_rk)):

    print(
        f"{x_rk[i]:.2f}\t "
        f"{y_rk[i]:.6f}\t "
        f"{exact_solution(x_rk[i]):.6f}\t "
        f"{error_exact_rk[i]:.10f}"
    )


# =====================================================
# ГРАФІКИ
# =====================================================

# -----------------------------
# Розв'язки
# -----------------------------
plt.figure(figsize=(10, 6))

x_exact = np.linspace(x0, xn, 500)

plt.plot(
    x_exact,
    exact_solution(x_exact),
    label="Точний розв'язок"
)

plt.plot(
    x_rk,
    y_rk,
    'o-',
    label="Рунге-Кутта 4"
)

plt.plot(
    x_adams,
    y_adams,
    's-',
    label="Адамс"
)

plt.grid()

plt.legend()

plt.title("Розв'язок диференціального рівняння")

plt.xlabel("x")

plt.ylabel("y")

plt.show()


# -----------------------------
# Похибка RK4
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    x_rk,
    error_exact_rk,
    'o-'
)

plt.grid()

plt.title("Локальна похибка RK4")

plt.xlabel("x")

plt.ylabel("error")

plt.show()


# -----------------------------
# Похибка Рунге
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    x_err,
    error_runge,
    'o-'
)

plt.grid()

plt.title("Оцінка похибки методом Рунге")

plt.xlabel("x")

plt.ylabel("error")

plt.show()


# -----------------------------
# Крок адаптації
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    x_auto[:-1],
    h_auto,
    'o-'
)

plt.grid()

plt.title("Автоматичний вибір кроку")

plt.xlabel("x")

plt.ylabel("h")

plt.show()