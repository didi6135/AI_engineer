import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data = pd.read_csv('ML_Data.csv')
#
# x, y = data['x1'], data['y']
# print(x)
x, y = make_regression(n_samples=100, n_features=4, noise=1, random_state=42)
plt.plot(x, y, 'o', color='blue')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = Ridge()
model.fit()
