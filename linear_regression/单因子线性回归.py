import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
data = pd.read_csv("linearData.csv")
print(data.head(), data.shape)


x = data.loc[:, 'x']
y = data.loc[:, 'y']

lr_model = LinearRegression()
x = np.array(x)
x = x.reshape(-1, 1)
y = np.array(y)
y = y.reshape(-1, 1)
lr_model.fit(x, y)

y_predict = lr_model.predict(x)
print(y_predict)


a = lr_model.coef_
b =lr_model.intercept_

print('a is', a,'b is', b)


#评估模型
MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y,y_predict)

print(MSE, R2)


plt.figure(figsize=(10, 10))
plt.scatter(y, y_predict)
plt.show()