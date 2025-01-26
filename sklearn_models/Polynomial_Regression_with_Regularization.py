import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# Generate synthetic dataset
np.random.seed(42)
X = np.random.uniform(1, 10, size=(50, 1))  # Hours studied
y = 3 * X ** 2 - 5 * X + 10 + np.random.normal(0, 5, size=(50, 1))  # Quadratic with noise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Polynomial degree
degree = 5

# Initialize models
models = {
    "Ridge": Ridge(alpha=1, max_iter=10000),
    "Lasso": Lasso(alpha=1, max_iter=10000),  # Adjust alpha for better convergence
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
}

# Prepare results storage
results = {}

# Plot the original data
plt.scatter(X, y, color="blue", alpha=0.6, label="True Data")
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

for name, model in models.items():
    # Create pipeline
    pipeline = make_pipeline(StandardScaler(),
                             PolynomialFeatures(degree, include_bias=False),
                             model)

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_range_pred = pipeline.predict(X_range)

    # Store results
    results[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

    # Plot regression curves
    plt.plot(X_range, y_range_pred, label=f"{name} Regression Curve")

# Finalize plot
plt.title(f"Polynomial Regression (Degree = {degree}) with Regularization")
plt.xlabel("Hours Studied")
plt.ylabel("Grades")
plt.legend()
plt.grid()
plt.show()

# Display results
print("Model Performance Comparison:")
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.2f}, MAE = {metrics['MAE']:.2f}")
