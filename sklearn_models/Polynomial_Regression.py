# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import PolynomialFeatures
#
# x = np.arange(0, 30)
# y = [3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50, 63, 67, 60, 62, 70, 75, 88, 81, 87, 95, 100, 108, 135, 151, 160, 169, 179]
#
#
# poly = PolynomialFeatures(degree=2, include_bias=False)
#
# x_poly = poly.fit_transform(x.reshape(-1, 1))
#
# model = LinearRegression()
# model.fit(x_poly, y)
#
# y_pred = model.predict(x_poly)
#
# mse = mean_squared_error(y, y_pred)
# mae = mean_absolute_error(y, y_pred)
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Mean Absolute Error (MAE): {mae}")
#
#
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.show()
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate synthetic data
np.random.seed(42)
X = np.linspace(1, 10, 100).reshape(-1, 1)  # Feature: 100 points from 1 to 10
y = 1000 * (X ** 2).flatten() + 5000 * X.flatten() + 10000 + np.random.normal(0, 20000, size=X.shape[0])  # Quadratic with reduced noise


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Polynomial degree
degree = 18

# Create polynomial features
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Train polynomial regression model
model = LinearRegression()
model.fit(X_train_poly_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_poly_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot true vs predicted values
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs True")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
# plt.title("True vs Predicted")
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.legend()
# plt.grid()
# plt.show()

# Visualize polynomial regression curve
X_full = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_full_poly = poly.transform(X_full)
X_full_poly_scaled = scaler.transform(X_full_poly)
y_full_pred = model.predict(X_full_poly_scaled)

plt.scatter(X, y, color='blue', alpha=0.6, label="Data Points")
plt.plot(X_full, y_full_pred, color='red', label="Polynomial Regression Line")
plt.title("Polynomial Regression Curve")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid()
plt.show()
