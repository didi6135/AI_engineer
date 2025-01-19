from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt



x, y = make_regression(n_samples=100, n_features=4, noise=10, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lasso_params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
}
lasso = Lasso()
lasso_model_cv = GridSearchCV(estimator=lasso,
                              param_grid=lasso_params,
                              n_jobs=-1,
                              cv=5)

lasso_model_cv.fit(x_train, y_train)

best_params = lasso_model_cv.best_params_
best_model = lasso_model_cv.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)

y_pred = best_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) on Test Data:", mse)

# Step 9: Visualize predictions vs true values
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Predictions vs True Values")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()