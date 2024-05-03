# 1.3
def simple_iteration_method(A, b, x0, e):
    n = len(A)
    x = x0.copy()
    x_new = [0] * n
    max_diff = float('inf')
    while max_diff > e:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        max_diff = 0
        for i in range(n):
            if abs(x_new[i] - x[i]) > max_diff:
                max_diff = abs(x_new[i] - x[i])
        x = x_new.copy()
    return x


def gauss_seidel_method(A, b, x0, e):
    n = len(A)
    x = x0.copy()
    max_diff = float('inf')
    while max_diff > e:
        for i in range(n):
            sigma1 = 0
            for j in range(i):
                sigma1 += A[i][j] * x[j]
            sigma2 = 0
            for j in range(i + 1, n):
                sigma2 += A[i][j] * x[j]
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i]
        x_new = x.copy()
        max_diff = 0
        for i in range(n):
            if abs(x_new[i] - x[i]) > max_diff:
                max_diff = abs(x_new[i] - x[i])
    return x


# Нач Усл
A = [
    [-14, 6, 1, -5],
    [-6, 27, 7, -6],
    [7, -5, -23, -8],
    [3, -8, -7, 26]
]
b = [95, -41, 69, 27]
x0 = [0] * len(b)
e = 0.01

solution_simple_iteration = simple_iteration_method(A, b, x0, e)
print("Метод простых итераций:", solution_simple_iteration)

solution_gauss_seidel = gauss_seidel_method(A, b, x0, e)
print("Метод Зейделя:", solution_gauss_seidel)

# 1.4
print()

import math


def rotation_matrix(A, i, j, theta):
    n = len(A)
    R = [[float(i == j) for j in range(n)] for i in range(n)]
    c = math.cos(theta)
    s = math.sin(theta)
    R[i][i] = c
    R[j][j] = c
    R[i][j] = -s
    R[j][i] = s
    return R


def max_off_diagonal_element(A):
    n = len(A)
    max_val = 0
    max_i = -1
    max_j = -1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_i = i
                max_j = j
    return max_i, max_j


def jacobi_rotation(A, e, max_iter=1000):
    n = len(A)
    eigenvectors = [[float(i == j) for j in range(n)] for i in range(n)]

    iter_count = 0
    while iter_count < max_iter:
        i, j = max_off_diagonal_element(A)
        if abs(A[i][j]) < e:
            break

        theta = 0.5 * math.atan2(2 * A[i][j], A[j][j] - A[i][i])

        R = rotation_matrix(A, i, j, theta)
        Rt = [[R[j][i] for j in range(n)] for i in range(n)]

        A = [[sum(R[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        A = [[sum(A[i][j] * Rt[i][k] for i in range(n)) for k in range(n)] for j in range(n)]

        eigenvectors = [[sum(eigenvectors[i][k] * R[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

        iter_count += 1

    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, eigenvectors


# Нач Усл
A = [
    [-3, -1, 3],
    [-1, 8, 1],
    [3, 1, 5]
]

e = 0.01

eigenvalues, eigenvectors = jacobi_rotation(A, e)

print("Собственные значения:", eigenvalues)
print("Собственные векторы:")
for i, vector in enumerate(eigenvectors):
    print(f"λ{i + 1} = {eigenvalues[i]}:", vector)

# 1.5
print()

A = [

    [1, 7, -1],
    [-2, 2, -2],
    [9, -7, 3]
]

x = [1, 1, 1]
max_iter = 1000
epsilon = 0.01

for i in range(max_iter):
    Ax = [sum(A[row][col] * x[col] for col in range(len(x))) for row in range(len(A))]

    max_element = max(Ax)

    x_new = [val / max_element for val in Ax]

    lambda_val = sum(x_new[i] / x[i] for i in range(len(x))) / len(x)

    if all(abs(x_new[i] - x[i]) < epsilon for i in range(len(x))):
        break

    x = x_new

spectral_radius = abs(lambda_val)
print("Спектральный радиус:", spectral_radius)

# 1.6
print()

import numpy as np


def qr_algorithm(A, epsilon):
    eigenvalues = np.linalg.eigvals(A)
    while True:
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)
        new_eigenvalues = np.diag(A)
        if np.max(np.abs(new_eigenvalues - eigenvalues)) < epsilon:
            break
        eigenvalues = new_eigenvalues
    return eigenvalues


A = np.array([
    [9, 0, 2],
    [-6, 4, 4],
    [-2, -7, 5]
])

epsilon = 0.01
eigenvalues = qr_algorithm(A, epsilon)
print("Собственные значения:", eigenvalues)
