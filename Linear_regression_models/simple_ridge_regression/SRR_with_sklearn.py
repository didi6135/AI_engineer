import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 100)
noise = np.random.normal(0, 5, 100)
raw_scores = 10 * hours_studied + 20 + noise
scores = np.clip(raw_scores, 0, 100)

x_train, x_test, y_train, y_test = train_test_split(hours_studied, scores, test_size=0.2)


model = Ridge()

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

model_cv.fit(x_train.reshape(-1, 1), y_train)
best_params = model_cv.best_params_
best_model = model_cv.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)

y_pred = best_model.predict(x_test.reshape(-1, 1))
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