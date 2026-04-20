import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Функція та точна похідна
# -----------------------------
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1
exact = dM(t0)

print("Точне значення похідної:", exact)


# -----------------------------
# 2. Центральна різниця
# -----------------------------
def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

hs = [0.1, 0.01, 0.001]

print("\nЧисельне диференціювання:")
for h in hs:
    approx = central_diff(M, t0, h)
    error = abs(approx - exact)
    print(f"h = {h}, D(h) = {approx:.6f}, похибка = {error:.6f}")


# -----------------------------
# 3. Вибір оптимального кроку
# -----------------------------
h_opt = 0.01
print("\nОптимальний крок h =", h_opt)


# -----------------------------
# 4. Два кроки
# -----------------------------
h1 = h_opt
h2 = h_opt / 2

D_h1 = central_diff(M, t0, h1)
D_h2 = central_diff(M, t0, h2)

print("\nДва кроки:")
print("D(h) =", D_h1)
print("D(h/2) =", D_h2)


# -----------------------------
# 5. Похибка
# -----------------------------
error_h = abs(D_h1 - exact)
print("\nПохибка при h:", error_h)


# -----------------------------
# 6. Метод Рунге-Ромберга
# -----------------------------
p = 2  # порядок центральної різниці

D_RR = D_h2 + (D_h2 - D_h1) / (2**p - 1)
error_RR = abs(D_RR - exact)

print("\nМетод Рунге-Ромберга:")
print("Уточнене значення:", D_RR)
print("Похибка:", error_RR)

h_values = np.logspace(-4, -1, 10)
errors = []

for h in h_values:
    approx = central_diff(M, t0, h)
    errors.append(abs(approx - exact))

plt.loglog(h_values, errors)
plt.xlabel("h")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кроку")
plt.grid()
plt.show()

# -----------------------------
# 7. Метод Ейткена
# -----------------------------
h3 = h_opt / 4

D_h3 = central_diff(M, t0, h3)

# Формула Ейткена
D_Aitken = D_h1 - ((D_h2 - D_h1)**2) / (D_h3 - 2*D_h2 + D_h1)
error_Aitken = abs(D_Aitken - exact)

# Оцінка порядку точності
p_est = np.log(abs((D_h3 - D_h2) / (D_h2 - D_h1))) / np.log(0.5)

print("\nМетод Ейткена:")
print("Уточнене значення:", D_Aitken)
print("Похибка:", error_Aitken)
print("Оцінка порядку точності p ≈", p_est)

# -----------------------------
# ОПТИМАЛЬНІ РЕЖИМИ ПОЛИВУ
# -----------------------------
k = np.arange(0, 5)
t_opt = np.pi + 2 * np.pi * k

print("\nОптимальні моменти поливу:")
for t in t_opt:
    print(f"t = {t:.2f}, M'(t) = {dM(t):.4f}")

# -----------------------------
# ГРАФІКИ
# -----------------------------
t = np.linspace(0, 20, 400)

plt.figure()

plt.plot(t, M(t), label="Вологість M(t)")
plt.plot(t, dM(t), label="Швидкість висихання M'(t)")

# позначаємо точки поливу
plt.scatter(t_opt, dM(t_opt), color='red', label="Полив (оптимум)")

plt.axhline(0, color='black', linewidth=0.5)

plt.title("Режими поливу рослини")
plt.xlabel("t")
plt.ylabel("Значення")
plt.legend()
plt.grid()

plt.show()