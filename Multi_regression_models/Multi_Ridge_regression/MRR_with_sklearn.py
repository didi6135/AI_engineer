import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

x, y = fetch_california_housing(return_X_y=True)
# plt.plot(x, y, 'o', color='blue')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

model = Ridge(max_iter=5000)
model.fit(x_train, y_train)



param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
}

model_cv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1

)


# Step 6: Fit GridSearchCV on training data
model_cv.fit(X_train, y_train)

# Step 7: Retrieve the best parameters and model
best_params = model_cv.best_params_
best_model = model_cv.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)

# Step 8: Evaluate the best model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) on Test Data:", mse)
print("Mean Absolute Error (MAE) on Test Data:", mae)

# Step 9: Visualize predictions vs true values
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Predictions vs True Values")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()
