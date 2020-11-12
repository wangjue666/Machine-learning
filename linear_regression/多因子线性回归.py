import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data = pd.read_csv("usa_housing_price.csv")

fig =  plt.figure(figsize=(10, 10))

fig1 = plt.subplot(231)
plt.scatter(data.loc[:,'Avg. Area Income'], data.loc[:,'Price'])
plt.title('Price vs Income')

fig2 = plt.subplot(232)
plt.scatter(data.loc[:,'Avg. Area House Age'], data.loc[:,'Price'])
plt.title('Price vs House Age')

fig3 = plt.subplot(233)
plt.scatter(data.loc[:,'Avg. Area Number of Rooms'], data.loc[:,'Price'])
plt.title('Price vs Number of Rooms')

fig4 = plt.subplot(234)
plt.scatter(data.loc[:,'Area Population'], data.loc[:,'Price'])
plt.title('Price vs Population')

fig5 = plt.subplot(235)
plt.scatter(data.loc[:,'Avg. Area Number of Bedrooms'], data.loc[:,'Price'])
plt.title('Price vs Number of Bedrooms')


#plt.show()

#define x and y

X = data.loc[:, 'Avg. Area Number of Bedrooms']
X = np.array(X).reshape(-1, 1)
y = data.loc[:, 'Price']
print(X.shape, y.shape)
#set up the linear regression model
LR1 = LinearRegression()
# train the model
LR1.fit(X, y)

#calculate the price vs size
y_predit_1 = LR1.predict(X)
#evaluate the model

mean_squared_error_1 = mean_squared_error(y, y_predit_1)
r2_score_1 = r2_score(y,y_predit_1)
print(mean_squared_error_1, r2_score_1)

fig6 = plt.figure(figsize=(8,5))
plt.scatter(X, y)
plt.plot(X,y_predit_1,'r')
#plt.show()

#define X_multi
X_multi = data.drop(['Price','Address'],axis=1)

#set up 2nd linear regression model
LR_multi = LinearRegression()
#train the model
LR_multi.fit(X_multi, y)

#make prediction
y_predict_multi = LR_multi.predict(X_multi)
mean_squared_error_multi = mean_squared_error(y, y_predict_multi)
r2_score_multi = r2_score(y,y_predict_multi)

print(mean_squared_error_multi, r2_score_multi)

fig7 = plt.figure(figsize=(8,5))

plt.scatter(y,y_predict_multi)
#plt.show()

x_test = np.array([65000,5,5,30000,200]).reshape(1,-1)

y_test_predict = LR_multi.predict(x_test)

print(x_test, y_test_predict, LR_multi.coef_)