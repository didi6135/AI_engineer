import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
X = np.linspace(1, 10, 100).reshape(-1, 1)  # Feature: 100 points from 1 to 10
y = 1000 * (X ** 2).flatten() + 5000 * X.flatten() + 10000 + np.random.normal(0, 20000, size=X.shape[0])  # Quadratic with reduced noise

# Plot the raw data
plt.scatter(X, y, color='blue', label="Data Points")
plt.title("Raw Data")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.grid()
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial degree
degree = 2

# Create polynomial features
X_train_poly = np.column_stack([X_train_scaled ** i for i in range(1, degree + 1)])
X_test_poly = np.column_stack([X_test_scaled ** i for i in range(1, degree + 1)])

# Define functions for polynomial regression
def predict(X, weights, bias):
    return X @ weights + bias

def error(X, y, weights, bias):
    return np.mean((y - predict(X, weights, bias)) ** 2)

def update_params(X, y, weights, bias, learning_rate):
    n = X.shape[0]
    y_pred = predict(X, weights, bias)
    derivative_weight = -(2 / n) * X.T @ (y - y_pred)
    derivative_bias = -(2 / n) * np.sum(y - y_pred)
    weights -= learning_rate * derivative_weight
    bias -= learning_rate * derivative_bias
    return weights, bias

def polynomial_regression(X, y, learning_rate, epochs):
    weights = np.zeros(X.shape[1])
    bias = 0
    losses = []
    for _ in range(epochs):
        weights, bias = update_params(X, y, weights, bias, learning_rate)
        losses.append(error(X, y, weights, bias))
    return weights, bias, losses

# Train the model
learning_rate = 0.1
epochs = 1000
weights, bias, losses = polynomial_regression(X_train_poly, y_train, learning_rate, epochs)

# Plot loss over epochs
plt.plot(losses, label="Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Predict on test data
y_pred = predict(X_test_poly, weights, bias)



# Visualize the polynomial regression curve
X_full = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_full_scaled = scaler.transform(X_full)
X_full_poly = np.column_stack([X_full_scaled ** i for i in range(1, degree + 1)])
y_full_pred = predict(X_full_poly, weights, bias)

plt.scatter(X, y, color='blue', label="Data Points")
plt.plot(X_full, y_full_pred, color='red', label="Polynomial Regression Line")
plt.title("Polynomial Regression Curve")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid()
plt.show()
