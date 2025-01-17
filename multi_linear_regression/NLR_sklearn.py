import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# load data from csv
df = pd.read_csv('Real-estate1.csv')
df.drop('No', inplace=True, axis=1)

# load data to plot
sns.lmplot(x='X4 number of convenience stores',
                y='Y house price of unit area', data=df)

plt.show()

# split to x and y
X_data = df.drop('Y house price of unit area',axis= 1)
Y_data = df['Y house price of unit area']

# split the all data to train and test
x_train, x_test, y_train, y_test = train_test_split(
    X_data, Y_data, test_size=0.3, random_state=101)

# create model and fit
model = LinearRegression()
model.fit(x_train, y_train)
print(f'weight: {model.coef_}, bias: {model.intercept_}')

# make prediction
predictions = model.predict(x_test)
print(f'MSE: {mean_squared_error(y_test, predictions)}')
print(f'MAE: {mean_absolute_error(y_test, predictions)}')