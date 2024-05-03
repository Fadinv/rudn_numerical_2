# 5.2 стр 93, вариант - з)


import numpy as np


def f(x):
    return pow(x[0], 4) + pow(x[1], 2) + pow(x[2], 2) + x[0] * x[1] + x[1] * x[2]


def pow(x, rate):
    return x ** rate


def gradient_f(x):
    df_x1 = 2 * x[0] * np.exp(x[0] ** 2) + 2 * (x[0] + x[1] + x[2])
    df_x2 = 2 * (x[0] + x[1] + x[2])
    df_x3 = 2 * (x[0] + x[1] + x[2])
    return np.array([df_x1, df_x2, df_x3])


x0 = np.array([0, 1, 0])
a = 0.638

x1 = x0 - a * gradient_f(x0)

fx0 = f(x0)
fx1 = f(x1)

print("5.2 стр 93, вариант - з)")
print("f(x^(0)): ", fx0)
print("f(x^(1)): ", fx1)

# 5.3 стр 97, вариант - г)
print()


def f(x):
    return 5 * (x[0] ** 2) - 4 * x[0] * x[1] + 5 * (x[1] ** 2) - x[0] - x[1]


def gradient_f(x):
    df_x1 = 8 * x[0] + 4 * x[1] - 17
    df_x2 = 4 * x[0] + 12 * x[1]
    return np.array([df_x1, df_x2])


x_k = np.array([0, 0])

epsilon = 0.01

# Метод наискорейшего спуска
while True:
    # Находим градиент в текущей точке
    grad_fx_k = gradient_f(x_k)

    # Проверяем условие остановки
    if np.abs(grad_fx_k[0]) <= epsilon and np.abs(grad_fx_k[1]) <= epsilon:
        break


    # Находим оптимальный шаг
    def g(a):
        return f(x_k - a * grad_fx_k)


    a_k = 0.1
    for _ in range(100):
        if g(a_k) < g(a_k + epsilon):
            break
        a_k += epsilon

    x_k1 = x_k - a_k * grad_fx_k

    x_k = x_k1

print("5.3 стр 97, вариант - г)")
print("Точка минимума x*:", x_k)
print("Значение функции в точке минимума f(x*):", f(x_k))

# 5.4 стр 101, вариант - е)
print()


def f(x):
    return 2 * (x[0] ** 2) - 2 * x[0] * x[1] + 3 * (x[1] ** 2) - 3 * x[1]


def gradient_f(x):
    df_x1 = 20 * x[0] + 3 * x[1]
    df_x2 = 3 * x[0] + 2 * x[1] + 10
    return np.array([df_x1, df_x2])


x_k = np.array([0, 0])

epsilon = 0.02

d_k = -gradient_f(x_k)

# Метод сопряженных градиентов
while True:
    def line_search(a):
        return f(x_k + a * d_k)


    a_k = 0.1
    for _ in range(100):
        if line_search(a_k) < line_search(
                a_k + epsilon):
            break
        a_k += epsilon

    x_k1 = x_k + a_k * d_k

    # Проверяем условие остановки
    if np.abs(gradient_f(x_k1)[0]) <= epsilon and np.abs(gradient_f(x_k1)[1]) <= epsilon:
        break

    grad_fx_k1 = gradient_f(x_k1)

    # Вычисляем коэффициент сопряженности
    beta_k = np.dot(grad_fx_k1, grad_fx_k1) / np.dot(gradient_f(x_k), gradient_f(x_k))

    # Обновляем направление
    d_k = -grad_fx_k1 + beta_k * d_k

    x_k = x_k1

print("5.4 стр 101, вариант - е)")
print("Точка минимума x*:", x_k)
print("Значение функции в точке минимума f(x*):", f(x_k))
