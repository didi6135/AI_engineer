from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, random_state=42, noise=10)

alpha = 0.1
rho = 0.5  # Balance between L1 and L2
learning_rate = 0.01
n_iterations = 1000


# Step 1: Define the ElasticNet loss function
def elasticnet_loss(X, y, w, alpha, rho):
    """
    Compute the ElasticNet loss: MSE + L1 + L2 penalties
    """
    n = len(y)  # Number of samples
    y_pred = X.dot(w)  # Predictions
    mse = np.mean(np.square(y - y_pred))  # Mean squared error
    l1_penalty = rho * alpha * np.sum(np.abs(w))  # L1 penalty
    l2_penalty = (1 - rho) * alpha * np.sum(np.square(w))  # L2 penalty
    return mse + l1_penalty + l2_penalty

# Step 2: Define the gradient for ElasticNet
def elasticnet_gradient(X, y, w, alpha, rho):
    """
    Compute the gradient of the ElasticNet loss function
    """
    n = len(y)  # Number of samples
    y_pred = X.dot(w)  # Predictions
    mse_gradient = -(2 / n) * X.T.dot(y - y_pred)  # Gradient of MSE
    l1_gradient = rho * alpha * np.sign(w)  # Gradient of L1 penalty
    l2_gradient = 2 * (1 - rho) * alpha * w  # Gradient of L2 penalty
    return mse_gradient + l1_gradient + l2_gradient

# Step 3: Implement gradient descent for ElasticNet Regression
def elasticnet_fit(X, y, alpha=0.1, rho=0.5, learning_rate=0.01, n_iterations=1000):
    """
    Perform ElasticNet regression using gradient descent
    """
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    loss_history = []
    for i in range(n_iterations):
        grad = elasticnet_gradient(X, y, w, alpha, rho)  # Compute gradient
        w -= learning_rate * grad  # Update weights
        loss = elasticnet_loss(X, y, w, alpha, rho)  # Compute loss
        loss_history.append(loss)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    return w, loss_history

# Step 4: Prediction function
def predict(X, w):
    return X.dot(w)


w_elasticnet, loss_history = elasticnet_fit(X, y, alpha, rho, learning_rate, n_iterations)
print("Final Weights:", w_elasticnet)

# Step 7: Visualize loss during training
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label="ElasticNet Loss")
plt.title("ElasticNet Loss During Training")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Step 8: Visualize predictions vs true values
y_pred = predict(X, w_elasticnet)
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7, color='blue', label="Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label="Ideal Prediction")
plt.title("Predictions vs True Values")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid()
plt.show()

# Step 9: Visualize weight updates
plt.figure(figsize=(10, 6))
plt.bar(range(len(w_elasticnet)), w_elasticnet, color="orange", label="Final Weights")
plt.title("ElasticNet Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.legend()
plt.grid()
plt.show()