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

x_original, y_original = read_data("data.csv")

x = x_original
y = y_original

print("x:", x)
print("y:", y)

def divided_differences_table(x, y):
    n = len(x)
    table = np.zeros((n, n))

    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table

table = divided_differences_table(x, y)

print("\nТаблиця розділених різниць:")
for i in range(len(x)):
    print(f"{x[i]:>6}", end=" ")
    for j in range(len(x)-i):
        print(f"{table[i][j]:>10.3f}", end=" ")
    print()

coef = divided_differences(x, y)

# Інтерполяція в межах вузлів!
rps_test = 600
cpu_600 = newton_polynomial(x, coef, rps_test)

# ===============================
# Основний графік
# ===============================
x_plot = np.linspace(min(x), max(x), 200)
y_plot = [newton_polynomial(x, coef, xi) for xi in x_plot]

plt.figure()

plt.scatter(x, y, color = 'red', label='Вузли')
plt.plot(x_plot, y_plot, label='Поліном Ньютона')

# точка прогнозу
plt.scatter(rps_test, cpu_600, color = 'orange', label='CPU при 600 RPS')
plt.text(rps_test, cpu_600, f'({rps_test},{cpu_600:.2f})')

plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.title("CPU = f(RPS)")
plt.legend()
plt.grid()
plt.show()

# ===============================
# Графіки для 10 і 20 вузлів
# ===============================

def increase_nodes(x, y, n_points):

    x_new = np.linspace(min(x), max(x), n_points)
    y_new = np.interp(x_new, x, y)

    return x_new, y_new

x10, y10 = increase_nodes(x, y, 10)
coef = divided_differences(x10, y10)
cpu_600_10 = newton_polynomial(x10, coef, rps_test)

x_plot = np.linspace(min(x10), max(x10), 200)
y_plot = [newton_polynomial(x10, coef, xi) for xi in x_plot]

plt.figure()

plt.scatter(x10, y10, color = 'red', label='Вузли')
plt.plot(x_plot, y_plot, label='Поліном Ньютона')

# точка прогнозу
plt.scatter(rps_test, cpu_600_10, color = 'orange', label='CPU при 600 RPS')
plt.text(rps_test, cpu_600_10, f'({rps_test},{cpu_600_10:.2f})')

plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.title("CPU = f(RPS)")
plt.legend()
plt.grid()
plt.show()

x20, y20 = increase_nodes(x, y, 20)
coef = divided_differences(x20, y20)
cpu_600_20 = newton_polynomial(x20, coef, rps_test)

x_plot = np.linspace(min(x20), max(x20), 200)
y_plot = [newton_polynomial(x20, coef, xi) for xi in x_plot]

plt.figure()

plt.scatter(x20, y20, color = 'red', label='Вузли')
plt.plot(x_plot, y_plot, label='Поліном Ньютона')

# точка прогнозу
plt.scatter(rps_test, cpu_600_20, color = 'orange', label='CPU при 600 RPS')
plt.text(rps_test, cpu_600_20, f'({rps_test},{cpu_600_20:.2f})')

plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.title("CPU = f(RPS)")
plt.legend()
plt.grid()
plt.show()

# ===============================
# Графік похибки
# ===============================

nodes_list = [5, 10, 20]
errors = [abs(cpu_600_20 - cpu_600) / 100,abs(cpu_600_20 - cpu_600_10) / 100,abs(cpu_600_20 - cpu_600_20) / 100]

# -------------------------------
# Побудова графіка похибки
# -------------------------------
plt.figure()

plt.plot(nodes_list, errors, marker='o')

plt.xlabel("Кількість вузлів")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кількості вузлів")

plt.grid()
plt.show()