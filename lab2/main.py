import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['rps']))
            y.append(float(row['cpu']))
    return np.array(x, dtype=float), np.array(y, dtype=float)


# ===============================
# Таблиця розділених різниць
# ===============================
def divided_differences(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

    return coef

# ===============================
# Поліном Ньютона (схема Горнера)
# ===============================
def newton_polynomial(x_data, coef, x):
    n = len(coef)
    p = coef[-1]
    for k in range(2, n + 1):
        p = coef[-k] + (x - x_data[-k]) * p
    return p

# ===============================
# Основна частина
# ===============================
x, y = read_data("data.csv")

print("x:", x)
print("y:", y)

coef = divided_differences(x, y)

# Інтерполяція в межах вузлів!
rps_test = 600
cpu_600 = newton_polynomial(x, coef, rps_test)

print("Значення полінома при 600 RPS =", cpu_600)

# ===============================
# Графік
# ===============================
x_plot = np.linspace(min(x), max(x) )
y_plot = [newton_polynomial(x, coef, xi) for xi in x_plot]

plt.scatter(x, y, color='red', label='Вузли')
plt.plot(x_plot, y_plot, label='Поліном Ньютона')
plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.legend()
plt.grid()
plt.show()