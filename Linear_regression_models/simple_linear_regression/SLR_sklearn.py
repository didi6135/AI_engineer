import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 200).reshape(-1, 1)
noise = np.random.normal(0, 5, 200).reshape(-1, 1)
raw_scores = 10 * hours_studied + 20 + noise
scores = np.clip(raw_scores, 0, 200)

x_train, x_test, y_train, y_test = train_test_split(hours_studied, scores, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

print(f"Weight (Slope): {model.coef_[0][0]:.2f}")
print(f"Bias (Intercept): {model.intercept_[0]:.2f}")

y_pred_test = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color="blue", label="True Scores")
plt.scatter(x_test, y_pred_test, color="red", label="Predicted Scores")
plt.plot(hours_studied, model.predict(hours_studied), color="green", linestyle="--", label="Regression Line")
plt.title("True vs Predicted Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.legend()
plt.grid()
plt.show()
