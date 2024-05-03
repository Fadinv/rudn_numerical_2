def f(x, y, A1, A2, A3):
    return A2 - A1 * y - (y ** 2) * (x - A3)


# Улучшенны метод Эйлера
def improved_euler_method(h, A1, A2, A3):
    x_values = [0]
    y_values = [1.7]

    x = 0
    y = 0.4

    while x < 1:
        x_new = x + h
        y_predictor = y + h * f(x, y, A1, A2, A3)
        y_corrector = y + h * (f(x, y, A1, A2, A3) + f(x_new, y_predictor, A1, A2, A3)) / 2
        y = y_corrector

        x = x_new
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values


# Вывод
A1 = 1.1
A2 = 1
A3 = 0.25
h = 0.1

x_values, y_values = improved_euler_method(h, A1, A2, A3)

for x, y in zip(x_values, y_values):
    print(f"x = {x:.2f}, y = {y:.6f}")
