import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Функція
# -----------------------------
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24


# -----------------------------
# 2. Точний інтеграл (наближено дуже точно)
# -----------------------------
def exact_integral():
    x = np.linspace(a, b, 100000)
    y = f(x)
    def trapz_manual(x, y):
        return np.sum((y[:-1] + y[1:]) * (x[1:] - x[:-1]) / 2)
    return trapz_manual(y, x)

I_exact = exact_integral()

# -----------------------------
# 3. Складова формула Сімпсона
# -----------------------------
def simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n має бути парним!")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    S = y[0] + y[-1]
    S += 4 * np.sum(y[1:-1:2])
    S += 2 * np.sum(y[2:-2:2])

    return S * h / 3


# -----------------------------
# 4. Дослідження похибки
# -----------------------------
n_values = [2, 4, 8, 16, 32, 64, 128]
errors = []
for n in n_values:
    I = simpson(f, a, b, n)
    errors.append(abs(I - I_exact))

plt.figure()
plt.loglog(n_values, errors, marker='o', label = "Сімпсон")
plt.xlabel("n")
plt.ylabel("Похибка")
plt.title("Залежність похибки від n (Сімпсон)")
plt.grid()
plt.show()


# -----------------------------
# 5. Рунге-Ромберг
# -----------------------------
RR_errors = []
for n in n_values:
    I_n = simpson(f, a, b, n)
    I_2n = simpson(f, a, b, 2*n)
    p = 4
    I_RR = I_2n + (I_2n - I_n) / (2**p - 1)
    RR_errors.append(abs(I_RR - I_exact))

plt.figure()
plt.loglog(n_values, RR_errors, marker='o', label="Рунге-Ронберг")
plt.xlabel("n")
plt.ylabel("Похибка")
plt.title("Залежність похибки від 2n (Рунге-Ронберг)")
plt.grid()
plt.show()


# -----------------------------
# 6. Метод Ейткена
# -----------------------------
A_errors = []
for n in n_values:
    I_n = simpson(f, a, b, n)
    I_2n = simpson(f, a, b, 2*n)
    I_4n = simpson(f, a, b, 4*n)
    I_A = I_n - (I_2n - I_n) ** 2 / (I_4n - 2 * I_2n + I_n)
    A_errors.append(abs(I_A - I_exact))

plt.figure()
plt.loglog(n_values, A_errors, marker='o', label="Ейткен")
plt.xlabel("n")
plt.ylabel("Похибка")
plt.title("Залежність похибки від 4n (Ейткен)")
plt.grid()
plt.show()

# -----------------------------
# 7. Адаптивний Сімпсон
# -----------------------------
def adaptive_simpson(f, a, b, eps, depth=0, max_depth=20):
    c = (a + b) / 2

    S = simpson(f, a, b, 2)
    S1 = simpson(f, a, c, 2)
    S2 = simpson(f, c, b, 2)

    if depth >= max_depth:
        return S1 + S2

    if abs(S1 + S2 - S) < 15 * eps:
        return S1 + S2 + (S1 + S2 - S) / 15
    else:
        return (adaptive_simpson(f, a, c, eps/2, depth+1) +
                adaptive_simpson(f, c, b, eps/2, depth+1))

eps_values = [1e-2, 1e-3, 1e-4, 1e-5]
adapt_errors = []

for eps in eps_values:
    I_adapt = adaptive_simpson(f, a, b, eps)
    adapt_errors.append(abs(I_adapt - I_exact))

plt.figure()
plt.loglog(eps_values, adapt_errors, marker='o')

plt.xlabel("eps")
plt.ylabel("Похибка")
plt.title("Похибка адаптивного алгоритму")
plt.grid()
plt.show()