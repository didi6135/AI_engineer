import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
n_samples, n_features = 100, 3

# Input features (X)
X = np.random.rand(n_samples, n_features)

# True weights and bias
true_weights = np.array([3, -2, 1.5])
true_bias = 5

# Target variable (y) with noise
y = X @ true_weights + true_bias + np.random.normal(0, 0.5, size=n_samples)

def predict(matrix, weight, bias):
    return matrix @ weight + bias

def error_func(matrix, y, weight, bias, lambda_reg):
    y_predict = predict(matrix, weight, bias)
    mse = np.mean((y - y_predict) ** 2)
    ridge_penalty = lambda_reg * np.sum(weight ** 2)
    return mse + ridge_penalty


def update_params(matrix, y, weight, bias, LR, lambda_reg):
    n = matrix.shape[0]
    y_predict = predict(matrix, weight, bias)

    # gradiant
    derivative_weight = -(2/n) * np.sum(matrix.T @ (y - y_predict)) + 2 * lambda_reg * weight
    derivative_bias = - (2/n) * np.sum(y - y_predict)

    weight -= derivative_weight * LR
    bias -= derivative_bias * LR

    return weight, bias


def ridge_regression(matrix, y, lambda_reg=10, learning_rate=0.001, epochs=1000):

    length_futures = matrix.shape[1]
    errors = []
    weight = np.zeros(length_futures)
    bias = 0

    for _ in range(epochs):
        weight, bias = update_params(matrix, y, weight, bias, learning_rate, lambda_reg)
        error = error_func(matrix, y, weight, bias, lambda_reg)
        errors.append(error)

    return weight, bias, errors



w_final, b_final, loss_history = ridge_regression(X, y)

# Print final weights and bias
print("Final Weights:", w_final)
print("Final Bias:", b_final)

# Plot loss over epochs
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Ridge Loss Over Time")
plt.show()
