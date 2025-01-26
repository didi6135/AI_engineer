import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


def lasso_loss(matrix, y, weight, alpha):
    y_predict = matrix.dot(weight)
    mse_error = np.mean(np.square(y - y_predict))
    l1_penalty = alpha * np.sum(np.abs(weight))
    return mse_error * l1_penalty


def lasso_gradiant(matrix, y, weight, alpha):
    length = len(y)
    y_predict = matrix.dot(weight)

    gradiant = -(2/length) * matrix.T.dot(y - y_predict)
    gradiant += alpha * np.sign(weight)
    return gradiant

def lasso(matrix, y, alpha, learning_rate, epochs):

    weights = np.zeros(matrix.shape[1])
    loss_history = []
    for i in range(epochs):
        gradiant = lasso_gradiant(matrix, y, weights, alpha)
        weights -= learning_rate * gradiant
        loss_func = lasso_loss(matrix, y, weights, alpha)
        loss_history.append(loss_func)
        if epochs % 100 == 0:
            print(f'epoch number: {i}, loss: {loss_func:.4f}')

    return weights, loss_history


def predict(matrix, weights):
    return matrix.dot(weights)

X, y = make_regression(n_samples=10, n_features=4, noise=10, random_state=42)
alpha = 0.1
learning_rate = 0.01
epochs = 1000

weight_lasso, loss_h = lasso(X, y, alpha, learning_rate, epochs)

plt.figure(figsize=(10, 6))
plt.plot(loss_h, label="Lasso Loss")
plt.title("Lasso Loss During Training")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Step 8: Visualize predictions vs true values
y_pred = predict(X, weight_lasso)
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
plt.bar(range(len(weight_lasso)), weight_lasso, color="orange", label="Final Weights")
plt.title("Lasso Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.legend()
plt.grid()
plt.show()