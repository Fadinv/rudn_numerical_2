# НУ
x0 = 1
e_const = 2.71828
y0 = 1.367879
x_end = 2
h = 0.1

# Дифференциальное уравнение
def f(x, y):
    return (y / (x ** 2) + e_const ** (x - 1 / x))

# Метод Эйлера
def euler_method(x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]
        y_next = y + h * f(x, y)
        x_values.append(x + h)
        y_values.append(y_next)
    return x_values, y_values

# Решение Эйлером
x_values, y_values = euler_method(x0, y0, x_end, h)

# Вывод
print("Эйлер")
print("x       y")
for x, y in zip(x_values, y_values):
    print(f"{x:.1f}     {y:.6f}")


# Неявный метод Эйлера
def implicit_euler_method(x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]
        y_next = y
        for _ in range(5):
            y_next = y + h * f(x + h, y_next)
        x_values.append(x + h)
        y_values.append(y_next)
    return x_values, y_values

# Решение неявным методом Эйлера
x_values_implicit, y_values_implicit = implicit_euler_method(x0, y0, x_end, h)

# Вывод
print("")
print("Неявным методом Эйлера")
print("x       y")
for x, y in zip(x_values_implicit, y_values_implicit):
    print(f"{x:.1f}     {y:.6f}")


# Метод Эйлера-Коши с итерационной обработкой
def euler_cauchy_method(x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]
        y_temp = y + h * f(x, y)
        for _ in range(5):
            y_next = y + h * (f(x, y) + f(x + h, y_temp)) / 2
            y_temp = y_next
        x_values.append(x + h)
        y_values.append(y_next)
    return x_values, y_values

# Решение методом Эйлера-Коши с итерационной обработкой
x_values_cauchy, y_values_cauchy = euler_cauchy_method(x0, y0, x_end, h)

# Вывод
print("")
print("Метод Эйлера-Коши с итерационной обработкой")
print("x       y")
for x, y in zip(x_values_cauchy, y_values_cauchy):
    print(f"{x:.1f}     {y:.6f}")


# Улучшенный метод Эйлера
def improved_euler_method(x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]
        y_temp = y + h * f(x, y)
        y_next = y + h * (f(x, y) + f(x + h, y_temp)) / 2
        x_values.append(x + h)
        y_values.append(y_next)
    return x_values, y_values

# Решение улучшенным методом Эйлера
x_values_improved, y_values_improved = improved_euler_method(x0, y0, x_end, h)

# Вывод
print("")
print("x       y")
for x, y in zip(x_values_improved, y_values_improved):
    print(f"{x:.1f}     {y:.6f}")


# Проверка
import math

def true_solution(x):
    return x * math.exp(1 / x)

print("")
print("Проверка")
print("x       y_true   y_euler     y_implicit  y_cauchy    y_improved")
for x, y_euler, y_implicit, y_cauchy, y_improved in zip(x_values, y_values, y_values_implicit, y_values_cauchy, y_values_improved):
    y_true = true_solution(x)
    print(f"{x:.1f}     {y_true:.6f}   {y_euler:.6f}   {y_implicit:.6f}   {y_cauchy:.6f}   {y_improved:.6f}")
