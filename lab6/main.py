import numpy as np

# =========================
# 1. Генерація даних
# =========================
def generate_matrix(n):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i][i] += n  # робить матрицю стійкою
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
# 2. LU-розклад (Дулітл)
# =========================
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1

    for k in range(n):
        # Обчислення L
        for i in range(k, n):
            sum_L = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_L

        # Обчислення U
        for j in range(k + 1, n):
            sum_U = sum(L[k][i] * U[i][j] for i in range(k))
            U[k][j] = (A[k][j] - sum_U) / L[k][k]

    return L, U

# =========================
# 3. Розв'язання LU
# =========================
def forward_substitution(L, b):
    n = len(b)
    z = np.zeros(n)

    for i in range(n):
        sum_z = sum(L[i][j] * z[j] for j in range(i))
        z[i] = (b[i] - sum_z) / L[i][i]

    return z

def backward_substitution(U, z):
    n = len(z)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_x = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = z[i] - sum_x

    return x

def solve_lu(L, U, b):
    z = forward_substitution(L, b)
    x = backward_substitution(U, z)
    return x

# =========================
# 4. Норми і похибка
# =========================
def norm(v):
    return np.max(np.abs(v))

def compute_residual(A, x, b):
    return b - A @ x

def compute_error(A, x, b):
    return norm(A @ x - b)

# =========================
# 5. Ітераційне уточнення
# =========================
def iterative_refinement(A, L, U, b, x0, eps=1e-14):
    x = x0.copy()
    iterations = 0

    while True:
        r = compute_residual(A, x, b)

        if norm(r) < eps:
            break

        delta_x = solve_lu(L, U, r)
        x = x + delta_x

        iterations += 1

        if norm(delta_x) < eps:
            break

    return x, iterations

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

# LU-розклад
L, U = lu_decomposition(A)

save_matrix("L.txt", L)
save_matrix("U.txt", U)

# Розв'язання
x = solve_lu(L, U, b)

# Похибка
eps = compute_error(A, x, b)
print("Похибка:", eps)

# Уточнення
x_refined, iters = iterative_refinement(A, L, U, b, x)

print("Кількість ітерацій:", iters)
print("Фінальна похибка:", compute_error(A, x_refined, b))

# порівняння з точним розв'язком
print("Відхилення від істинного x:",
      norm(x_refined - x_true))
print("\n===== РОЗВ'ЯЗОК СЛАР (x) =====")
print(x_refined)