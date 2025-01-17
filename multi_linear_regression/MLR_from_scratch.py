import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 100
n_dims = 3

X = np.random.rand(n_samples, n_dims)
weights_true = np.array([3, -2, 1.5])
bias_true = 5
y = X @ weights_true + bias_true + np.random.normal(0, 0.5, size=n_samples)

print(weights_true)

def predict(matrix, weight, bias):
    return matrix @ weight + bias


def compute_mse(y, y_predict):
    return np.mean((y - y_predict) ** 2)



def update_params(matrix, y_values, weight, bias, LR):
    n = matrix.shape[0]
    y_predict = predict(matrix, weight, bias)

    derivative_weight = -(2/n) * (matrix.T @ (y_values - y_predict))
    derivative_bias = -(2/n) * np.sum(y_values - y_predict)

    weight -= derivative_weight * LR
    bias -= derivative_bias * LR

    return weight, bias



def multi_linear_regression(X, y, learning_rate=0.001, epochs=1000):
    n = X.shape[1]
    w = np.zeros(n)
    b = 0
    errors = []

    for _ in range(epochs):
        w, b = update_params(X, y, w, b, learning_rate)
        y_predict = predict(X, w, b)
        err = compute_mse(y, y_predict)
        errors.append(err)

    return w, b, errors


w_final, b_final, mse_history = multi_linear_regression(X, y)

print("משקלים סופיים:", w_final)
print("חיתוך סופי:", b_final)

plt.plot(mse_history)
plt.xlabel('מחזורים')
plt.ylabel('MSE')
plt.title('ירידת השגיאה לאורך הזמן')
plt.show()
