import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave

X, Y = make_wave()
line = np.linspace(-3, 3, 100).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(f'weight: {model.coef_}, bias: {model.intercept_}')

plt.plot(x_train, y_train, 'o', color='blue')
plt.plot(line, model.predict(line))
# plt.plot(x_test, y_test, '.', color='red')
plt.show()
