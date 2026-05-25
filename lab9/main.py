import math
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# ЦІЛЬОВІ ФУНКЦІЇ
# ==========================================

# 1. Функція Розенброка
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


# 2. Степенева функція
def power_function(x):
    return abs(x[0])**2 + abs(x[1])**3


# 3. Коренева функція
def root_function(x):
    return math.sqrt(abs(x[0])) + math.sqrt(abs(x[1]))


# ==========================================
# ПРИКЛАД СИСТЕМИ НЕЛІНІЙНИХ РІВНЯНЬ
# ==========================================
# x^2 + y^2 - 4 = 0
# x - y - 1 = 0

def f1(x):
    return x[0]**2 + x[1]**2 - 4


def f2(x):
    return x[0] - x[1] - 1


# Цільова функція для системи
def system_target(x):
    return f1(x)**2 + f2(x)**2


# ==========================================
# МЕТОД ХУКА-ДЖИВСА
# ==========================================

def exploratory_search(func, base_point, step):
    """
    Досліджуючий пошук
    """
    x = np.array(base_point, dtype=float)
    n = len(x)

    for i in range(n):

        # Рух у додатному напрямку
        x_forward = np.copy(x)
        x_forward[i] += step[i]

        if func(x_forward) < func(x):
            x = x_forward
        else:
            # Рух у від’ємному напрямку
            x_backward = np.copy(x)
            x_backward[i] -= step[i]

            if func(x_backward) < func(x):
                x = x_backward

    return x


def hooke_jeeves(func,
                 x0,
                 step_size=0.5,
                 alpha=2.0,
                 epsilon=1e-6,
                 max_iter=1000):

    """
    Метод Хука-Дживса
    """

    n = len(x0)

    base_point = np.array(x0, dtype=float)
    step = np.array([step_size] * n)

    iterations = 0

    print("===================================")
    print("ПОЧАТОК РОБОТИ МЕТОДУ")
    print("===================================")

    while np.max(step) > epsilon and iterations < max_iter:

        iterations += 1

        print(f"\nІтерація {iterations}")
        print("Базисна точка:", base_point)
        print("Значення функції:", func(base_point))

        # Досліджуючий пошук
        new_point = exploratory_search(func, base_point, step)

        if func(new_point) < func(base_point):

            # Пошук по зразку
            while True:

                pattern_point = new_point + alpha * (new_point - base_point)

                explored_point = exploratory_search(
                    func,
                    pattern_point,
                    step
                )

                if func(explored_point) < func(new_point):
                    base_point = new_point
                    new_point = explored_point
                else:
                    break

            base_point = new_point

        else:
            # Зменшення кроку
            step = step / 2.0

            print("Крок зменшено:", step)

    print("\n===================================")
    print("МІНІМУМ ЗНАЙДЕНО")
    print("===================================")

    print("Точка мінімуму:", base_point)
    print("Значення функції:", func(base_point))
    print("Кількість ітерацій:", iterations)

    return base_point


# ==========================================
# ПОБУДОВА ГРАФІКІВ СИСТЕМИ
# ==========================================

def plot_system():

    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)

    X, Y = np.meshgrid(x, y)

    Z1 = X**2 + Y**2 - 4
    Z2 = X - Y - 1

    plt.figure(figsize=(8, 8))

    plt.contour(X, Y, Z1, levels=[0], colors='blue')
    plt.contour(X, Y, Z2, levels=[0], colors='red')

    plt.title("Система нелінійних рівнянь")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.axis('equal')

    plt.show()


# ==========================================
# ГОЛОВНА ПРОГРАМА
# ==========================================

if __name__ == "__main__":

    # Побудова графіків
    plot_system()

    # Початкове наближення
    x0 = [1.0, 1.0]

    print("\n===================================")
    print("РОЗВ'ЯЗОК СИСТЕМИ")
    print("===================================")

    solution = hooke_jeeves(
        func=system_target,
        x0=x0,
        step_size=0.5,
        alpha=2.0,
        epsilon=1e-6
    )

    print("\nРозв'язок системи:")
    print("x =", solution[0])
    print("y =", solution[1])

    print("\nПеревірка:")
    print("f1 =", f1(solution))
    print("f2 =", f2(solution))

    # ======================================
    # ТЕСТ НА ФУНКЦІЇ РОЗЕНБРОКА
    # ======================================

    print("\n===================================")
    print("ТЕСТ ФУНКЦІЇ РОЗЕНБРОКА")
    print("===================================")

    x0_rosen = [-1.2, 1]

    minimum = hooke_jeeves(
        func=rosenbrock,
        x0=x0_rosen,
        step_size=0.5,
        alpha=2.0,
        epsilon=1e-6
    )

    print("\nМінімум функції Розенброка:")
    print("x =", minimum[0])
    print("y =", minimum[1])