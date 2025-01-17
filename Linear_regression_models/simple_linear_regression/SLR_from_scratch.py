import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(-3, 3, 100)
y = 3 * x + 7 + np.random.normal(0, 3, size=len(x))


def error_function(x_now, y_now, weight, bias):
    y_predict = weight * x_now + bias
    mean_square_error = np.mean((y_now - y_predict) ** 2)
    return mean_square_error


def update_params(weight_now, bias_now, x_points, y_points, LR):
    n = len(x)
    y_predict = weight_now * x_points + bias_now

    derivative_weight = -(2/n) * np.sum(x_points * (y_points - y_predict))
    derivative_bias = -(2/n) * np.sum(y_points - y_predict)

    weight_now -= derivative_weight * LR
    bias_now -= derivative_bias * LR
    print(f'dw: {weight_now}, db: {bias_now}')
    return weight_now, bias_now


def linear_regression(x_points, y_points, learning_rate=0.0001, epochs=1):
    weight, bias = 0, 0
    errors = []

    for _ in range(epochs):
        weight, bias = update_params(weight, bias, x_points, y_points, learning_rate)
        error = error_function(x_points, y_points, weight, bias)
        errors.append(error)

    return weight, bias, errors


weight, bias, errors = linear_regression(x, y)

plt.plot(errors)
plt.show()

plt.scatter(x, y, color='blue')
plt.plot(x, weight * x + bias, color='red')
plt.show()