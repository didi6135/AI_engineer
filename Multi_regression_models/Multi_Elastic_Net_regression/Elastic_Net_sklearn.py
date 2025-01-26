from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

tips = sns.load_dataset(name='tips')
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
#

tips = pd.get_dummies(tips, dtype=float)
x = tips.drop('tip', axis=1)
y = tips['tip']

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = ElasticNet(max_iter=5000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


print(f'mse: {mean_squared_error(y_test, y_pred)}')

elastic_params = {
    'alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7],
    'l1_ratio': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
}

elastic_cv = GridSearchCV(model, elastic_params, n_jobs=-1, cv=3)
elastic_cv.fit(x_train, y_train)
y_pred2 = elastic_cv.predict(x_test)

best_params = elastic_cv.best_params_
best_model = elastic_cv.best_estimator_

print(f'best model is: {best_model}')
print(f'mse: {mean_squared_error(y_test, y_pred2)}')
