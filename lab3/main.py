import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Зчитування даних з CSV
# -------------------------------
def read_csv(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return np.array(x), np.array(y)


# -------------------------------
# 2. Формування матриці A
# -------------------------------
def form_matrix(x, m):
    n = m + 1
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i][j] = np.sum(x ** (i + j))

    return A


# -------------------------------
# 3. Формування вектора b
# -------------------------------
def form_vector(x, y, m):
    n = m + 1
    b = np.zeros(n)

    for i in range(n):
        b[i] = np.sum(y * (x ** i))

    return b


# -------------------------------
# 4. Метод Гауса
# -------------------------------
def gauss_solve(A, b):
    n = len(b)

    for k in range(n):
        # вибір головного елемента
        max_row = np.argmax(abs(A[k:, k])) + k

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    # зворотний хід
    x_sol = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i+1:] * x_sol[i+1:])) / A[i, i]

    return x_sol


# -------------------------------
# 5. Поліном
# -------------------------------
def polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)

    for i in range(len(coef)):
        y += coef[i] * (x ** i)

    return y


# -------------------------------
# 6. Дисперсія
# -------------------------------
def variance(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# 7. Основна програма
# -------------------------------
x, y = read_csv("temperature.csv")

max_degree = 6
variances = []

for m in range(1, max_degree + 1):

    A = form_matrix(x, m)
    b = form_vector(x, y, m)

    coef = gauss_solve(A.copy(), b.copy())

    y_approx = polynomial(x, coef)

    var = variance(y, y_approx)

    variances.append(var)

# оптимальний степінь
optimal_m = np.argmin(variances) + 1

print("Дисперсії:")
for i, v in enumerate(variances):
    print(f"m = {i+1}: {v}")

print("\nОптимальний степінь:", optimal_m)


# -------------------------------
# 8. Фінальна апроксимація
# -------------------------------
A = form_matrix(x, optimal_m)
b = form_vector(x, y, optimal_m)

coef = gauss_solve(A.copy(), b.copy())

y_approx = polynomial(x, coef)


# -------------------------------
# 9. Прогноз
# -------------------------------
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef)

print("\nПрогноз температур:")
for i in range(3):
    print(f"Month {x_future[i]}: {y_future[i]:.2f}")


# -------------------------------
# 10. Похибка
# -------------------------------
error = y - y_approx


# -------------------------------
# 11. Графік апроксимації
# -------------------------------
plt.figure()

plt.scatter(x, y, label="Реальні дані")
plt.plot(x, y_approx, label="Апроксимація")

plt.xlabel("Month")
plt.ylabel("Temperature")
plt.legend()
plt.title("Апроксимація методом найменших квадратів")

plt.show()


# -------------------------------
# 12. Графік дисперсії
# -------------------------------
plt.figure()

degrees = np.arange(1, max_degree + 1)
plt.plot(degrees, variances, marker='o')

plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")
plt.title("Залежність дисперсії від степеня")

plt.show()


# -------------------------------
# 13. Графік похибки
# -------------------------------
plt.figure()

plt.plot(x, error, marker='o')

plt.xlabel("Month")
plt.ylabel("Error")
plt.title("Похибка апроксимації")

plt.show()