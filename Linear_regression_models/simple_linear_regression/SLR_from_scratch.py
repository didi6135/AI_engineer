import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 100)
noise = np.random.normal(0, 5, 100)
raw_scores = 10 * hours_studied + 20 + noise
scores = np.clip(raw_scores, 0, 100)



def error_function(x_now, y_now, weight, bias):
    y_predict = weight * x_now + bias
    mean_square_error = np.mean((y_now - y_predict) ** 2)
    return mean_square_error


def update_params(weight_now, bias_now, x_points, y_points, LR):
    n = len(x_points)
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


learning_rate = 0.001
epochs = 2000
weight, bias, errors = linear_regression(hours_studied, scores, learning_rate, epochs)

plt.figure(figsize=(10, 6))
plt.plot(errors)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(hours_studied, scores, color="blue", alpha=0.6, label="Actual Data")
plt.plot(hours_studied, weight * hours_studied + bias, color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid()
plt.show()

print(f"Weight (Slope): {weight}")
print(f"Bias (Intercept): {bias}")


def predict(hours, weight, bias):
    return weight * hours + bias


np.random.seed(42)
test_hours = np.random.uniform(1, 10, 20)
true_scores = 10 * test_hours + 20 + np.random.normal(0, 5, 20)
true_scores = np.clip(true_scores, 0, 100)

predicted_scores = predict(test_hours, weight, bias)

mae = mean_absolute_error(true_scores, predicted_scores)
mse = mean_squared_error(true_scores, predicted_scores)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(test_hours, true_scores, color="blue", label="True Scores")
plt.scatter(test_hours, predicted_scores, color="red", label="Predicted Scores")
plt.plot(test_hours, predicted_scores, color="red", linestyle="--", alpha=0.7, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.title("True vs Predicted Exam Scores")
plt.legend()
plt.grid()
plt.show()