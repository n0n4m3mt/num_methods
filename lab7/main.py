import numpy as np

# =========================
# 1. Генерація матриці (діагональне переважання)
# =========================
def generate_matrix(n):
    A = np.random.rand(n, n)

    for i in range(n):
        A[i][i] = sum(abs(A[i])) + 1  # діагональне переважання

    return A


def generate_solution(n):
    return np.full(n, 2.5)


def compute_b(A, x):
    return A @ x


def save_matrix(filename, A):
    np.savetxt(filename, A)


def save_vector(filename, v):
    np.savetxt(filename, v)


def load_matrix(filename):
    return np.loadtxt(filename)


def load_vector(filename):
    return np.loadtxt(filename)


# =========================
# 2. Допоміжні функції
# =========================
def norm(v):
    return np.max(np.abs(v))


def matrix_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))


# =========================
# 3. Метод простої ітерації
# =========================
def simple_iteration(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(b)
    tau = 1.0 / matrix_norm(A)  # стабільний вибір

    C = np.eye(n) - tau * A
    d = tau * b

    x = x0.copy()

    for k in range(max_iter):
        x_new = C @ x + d

        if norm(x_new - x) < eps:
            return x_new, k + 1

        x = x_new

    return x, max_iter


# =========================
# 4. Метод Якобі
# =========================
def jacobi(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(b)
    x = x0.copy()

    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(max_iter):
        x_new = (b - R @ x) / D

        if norm(x_new - x) < eps:
            return x_new, k + 1

        x = x_new

    return x, max_iter


# =========================
# 5. Метод Зейделя
# =========================
def gauss_seidel(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])

            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]

        if norm(x_new - x) < eps:
            return x_new, k + 1

        x = x_new

    return x, max_iter


# =========================
# MAIN
# =========================
n = 100

# Генерація
A = generate_matrix(n)
x_true = generate_solution(n)
b = compute_b(A, x_true)

save_matrix("A.txt", A)
save_vector("B.txt", b)

# Завантаження
A = load_matrix("A.txt")
b = load_vector("B.txt")

# Початкове наближення
x0 = np.ones(n)

# =========================
# Обчислення
# =========================
x_si, it_si = simple_iteration(A, b, x0)
x_jacobi, it_j = jacobi(A, b, x0)
x_gs, it_gs = gauss_seidel(A, b, x0)

# =========================
# Вивід
# =========================
print("Метод простої ітерації:")
print("Ітерації:", it_si)
print("Похибка:", norm(A @ x_si - b))
print("Розв'язок:", x_si)

print("\nМетод Якобі:")
print("Ітерації:", it_j)
print("Похибка:", norm(A @ x_jacobi - b))
print("Розв'язок:", x_jacobi)

print("\nМетод Зейделя:")
print("Ітерації:", it_gs)
print("Похибка:", norm(A @ x_gs - b))
print("Розв'язок:", x_gs)