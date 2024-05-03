import numpy as np
import matplotlib.pyplot as plt


def explicit_scheme(U, x, t, k, h, cos_x, cos_t, sin_t):
    N = len(x)
    M = len(t)
    U_new = np.zeros((N, M))
    U_new[:, 0] = U[:, 0]
    for n in range(M - 1):
        for i in range(1, N - 1):
            U_new[i, n + 1] = U[i, n] + k / h ** 2 * (U[i + 1, n] - 2 * U[i, n] + U[i - 1, n]) + k * cos_x[i] * (
                    cos_t[n] + sin_t[n])
        U_new[0, n + 1] = sin_t[n]
        U_new[-1, n + 1] = -sin_t[n]
    return U_new


def implicit_scheme(U, x, t, k, h, cos_x, cos_t, sin_t):
    N = len(x)
    M = len(t)
    U_new = np.zeros((N, M))
    U_new[:, 0] = U[:, 0]
    A = np.zeros((N, N))
    b = np.zeros(N)
    for n in range(M - 1):
        for i in range(1, N - 1):
            A[i, i - 1] = -k / h ** 2
            A[i, i] = 1 + 2 * k / h ** 2
            A[i, i + 1] = -k / h ** 2
            b[i] = U[i, n] + k * cos_x[i] * (cos_t[n] + sin_t[n])
        A[0, 0] = 1
        b[0] = sin_t[n]
        A[-1, -1] = 1
        b[-1] = -sin_t[n]
        U_new[:, n + 1] = np.linalg.solve(A, b)
    return U_new


def crank_nicolson_scheme(U, x, t, k, h, cos_x, cos_t, sin_t):
    N = len(x)
    M = len(t)
    U_new = np.zeros((N, M))
    U_new[:, 0] = U[:, 0]
    A = np.zeros((N, N))
    b = np.zeros(N)
    for n in range(M - 1):
        for i in range(1, N - 1):
            A[i, i - 1] = -k / (2 * h ** 2)
            A[i, i] = 1 + k / h ** 2
            A[i, i + 1] = -k / (2 * h ** 2)
            b[i] = U[i, n] + k / (2 * h ** 2) * (U[i + 1, n] - 2 * U[i, n] + U[i - 1, n]) + k / 2 * cos_x[i] * (
                    cos_t[n] + sin_t[n])
        A[0, 0] = 1
        b[0] = sin_t[n]
        A[-1, -1] = 1
        b[-1] = -sin_t[n]
        U_new[:, n + 1] = np.linalg.solve(A, b)
    return U_new


def analytical_solution(x, t):
    return np.sin(t) * np.cos(x)


x = np.linspace(0, np.pi / 2, 100)
t = np.linspace(0, 10, 100)
k = t[1] - t[0]
h = x[1] - x[0]
cos_x = np.cos(x)
cos_t = np.cos(t)
sin_t = np.sin(t)

U_explicit = explicit_scheme(np.zeros((len(x), len(t))), x, t, k, h, cos_x, cos_t, sin_t)
U_implicit = implicit_scheme(np.zeros((len(x), len(t))), x, t, k, h, cos_x, cos_t, sin_t)
U_cn = crank_nicolson_scheme(np.zeros((len(x), len(t))), x, t, k, h, cos_x, cos_t, sin_t)

U_analytical = analytical_solution(x, t)

print("Метод Эйлера:")
print("Максимальная погрешность:", np.max(np.abs(U_explicit - U_analytical)))
print("Обратный метода Эйлера:")
print("Максимальная погрешность:", np.max(np.abs(U_implicit - U_analytical)))
print("Кранка-Никольсон:")
print("Максимальная погрешность:", np.max(np.abs(U_cn - U_analytical)))

# Plot the results
plt.plot(x, U_explicit[:, -1], label='Метод Эйлера')
plt.plot(x, U_implicit[:, -1], label='Обратный метода Эйлера')
plt.plot(x, U_cn[:, -1], label='Кранка-Никольсон')
plt.plot(x, U_analytical, label='Аналитическая функция')
plt.legend()
plt.show()
