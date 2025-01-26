import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate synthetic data: hours studied vs exam scores
np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 100)
noise = np.random.normal(0, 5, 100)
raw_scores = 10 * hours_studied + 20 + noise
scores = np.clip(raw_scores, 0, 100)




# Ridge Regression from Scratch for Single Feature

def predict_ridge(x, weight, bias):
    return weight * x + bias


def error_func_ridge(x, y, weight, bias, lambda_reg):
    y_pred = predict_ridge(x, weight, bias)
    mse = np.mean((y - y_pred) ** 2)
    ridge_penalty = lambda_reg * (weight ** 2)
    return mse + ridge_penalty


def update_params_ridge(x, y, weight, bias, learning_rate, lambda_reg):
    n = len(y)
    y_pred = predict_ridge(x, weight, bias)

    # Calculate gradients
    dw = -(2 / n) * np.sum(x * (y - y_pred)) + 2 * lambda_reg * weight
    db = -(2 / n) * np.sum(y - y_pred)

    # Update parameters
    weight -= learning_rate * dw
    bias -= learning_rate * db

    return weight, bias


def ridge_regression(x, y, lambda_reg=1, learning_rate=0.01, epochs=1000):
    weight, bias = 0, 0  # Initialize parameters
    errors = []

    for epoch in range(epochs):
        weight, bias = update_params_ridge(x, y, weight, bias, learning_rate, lambda_reg)
        loss = error_func_ridge(x, y, weight, bias, lambda_reg)
        errors.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return weight, bias, errors


# Train Ridge Regression
lambda_reg = 0.5
learning_rate = 0.01
epochs = 1000

w_final, b_final, loss_history = ridge_regression(hours_studied, scores, lambda_reg, learning_rate, epochs)

# Print final results
print(f"Final Weight (Slope): {w_final}")
print(f"Final Bias (Intercept): {b_final}")

# Plot loss over epochs
plt.plot(loss_history)
plt.title("Ridge Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Plot regression line
plt.scatter(hours_studied, scores, color="blue", label="True Scores")
predicted_scores = predict_ridge(hours_studied, w_final, b_final)
plt.plot(hours_studied, predicted_scores, color="red", label="Ridge Regression Line")
plt.title("True vs Predicted Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.legend()
plt.grid()
plt.show()



def predict(hours, weight, bias):
    return weight * hours + bias


np.random.seed(42)
test_hours = np.random.uniform(1, 10, 20)
true_scores = 10 * test_hours + 20 + np.random.normal(0, 5, 20)
true_scores = np.clip(true_scores, 0, 100)

predicted_scores = predict(test_hours, w_final, b_final)

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