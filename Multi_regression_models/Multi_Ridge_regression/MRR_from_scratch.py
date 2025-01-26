import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = fetch_california_housing(return_X_y=True)
y = y * 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



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


def ridge_regression_multi(matrix, y, lambda_reg=10, learning_rate=0.001, epochs=1000):

    length_futures = matrix.shape[1]
    errors = []
    weight = np.zeros(length_futures)
    bias = 0

    for _ in range(epochs):
        weight, bias = update_params(matrix, y, weight, bias, learning_rate, lambda_reg)
        error = error_func(matrix, y, weight, bias, lambda_reg)
        errors.append(error)

    return weight, bias, errors



# Train the Ridge Regression model
lambda_reg = 1  # Regularization parameter
learning_rate = 0.01
epochs = 2000

weights_final, bias_final, loss_history = ridge_regression_multi(X_train, y_train, lambda_reg, learning_rate, epochs)

# Print final weights and bias
print("Final Weights:", weights_final)
print("Final Bias:", bias_final)

# Plot loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Ridge Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Predict on the test set
y_pred = predict(X_test, weights_final, bias_final)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) on Test Data:", mse)
print("Mean Absolute Error (MAE) on Test Data:", mae)

# Scatter plot: true vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted vs True")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.title("True vs Predicted House Values")
plt.xlabel("True Median House Value")
plt.ylabel("Predicted Median House Value")
plt.legend()
plt.grid()
plt.show()

