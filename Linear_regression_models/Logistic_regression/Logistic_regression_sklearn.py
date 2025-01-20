import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

pima = pd.read_csv("diabetes.csv")


x = pima.drop('Outcome', axis=1)
y = pima['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=16)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = LogisticRegression(max_iter=5000, random_state=16)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# search the best combination
param_grid = {
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    'C': [0.01, 0.1, 1, 10, 100],
}

LR_cv = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
LR_cv.fit(x_train, y_train)
best_model = LR_cv.best_estimator_
best_params = LR_cv.best_params_

print(f'Best model is: {best_model}, best params is: {best_params}')

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic", "Diabetic"]).plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.6, label='True Labels')
plt.scatter(range(len(y_test)), y_pred, color='red', alpha=0.6, label='Predicted Labels')
plt.xlabel("Sample Index")
plt.ylabel("Outcome")
plt.title("True vs Predicted Outcomes")
plt.legend()
plt.grid()
plt.show()

# Step 9: Visualize feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.grid()
plt.show()