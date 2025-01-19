import numpy as np
import matplotlib.pyplot as plt
from scipy.differentiate import derivative


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss_func(y, y_predict):

    cross_entropy = -np.mean(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
    return cross_entropy



def update_params(matrix, y, weights, bias, learning_rate):

    length = len(y)
    z = matrix.dot(weights) + bias
    y_predict = sigmoid(z)

    derivative_weight = (1 / length) * np.dot(matrix.T, (y_predict - y))
    derivative_bias = (1 / length) * np.sum(y_predict - y)

    weights -= learning_rate * derivative_weight
    bias -= learning_rate * derivative_bias

    return  weights, bias, y_predict



def logistic_regression(matrix, y, learning_rate, epochs):
    weights = np.zeros(matrix.shape[1])
    bias = 0
    errors = []

    for i in range(epochs):
        weights, bias, y_predict = update_params(matrix, y, weights, bias, learning_rate)
        loss_func = cross_entropy_loss_func(y, y_predict)
        errors.append(loss_func)

        if i % 100 == 0:
            print(f'epoch number: {i}, error = {loss_func:.4f}')

    return weights, bias, errors

def predict(X, w, b, threshold=0.5):
    """
    Predict class labels based on probabilities.
    """
    probabilities = sigmoid(np.dot(X, w) + b)
    return (probabilities >= threshold).astype(int)


np.random.seed(42)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Hours studied
y = np.array([0, 0, 0, 1, 1, 1])  # Pass or fail

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std


learning_rate = 0.1
n_iterations = 1000

w, b, loss_history = logistic_regression(X, y, learning_rate, n_iterations)
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Step 8: Visualize predictions
X_test = np.linspace(-2, 2, 100).reshape(-1, 1)  # Test data
y_prob = sigmoid(np.dot(X_test, w) + b)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Training Data")
plt.plot(X_test, y_prob, color="red", label="Sigmoid Curve")
plt.title("Logistic Regression Sigmoid Fit")
plt.xlabel("Hours Studied (Standardized)")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid()
plt.show()

# Step 9: Make predictions on the original dataset
y_pred = predict(X, w, b)
print("Predicted labels:", y_pred)