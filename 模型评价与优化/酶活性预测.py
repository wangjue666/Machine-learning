import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
data_train = pd.read_csv('./T-R-train.csv')


#define x_train AND y_train
X_train = data_train.loc[:,"T"]
y_train = data_train.loc[:,'rate']

#visualize the data
fig1 = plt.figure(figsize=(5,5))

plt.scatter(X_train, y_train)
plt.title('raw data')
plt.xlabel('temperature')
plt.ylabel('rate')
#plt.show()

#linear regression model prediction
X_train = np.array(X_train).reshape(-1, 1)
lr1 = LinearRegression()
lr1.fit(X_train, y_train)

#load the test data

data_test = pd.read_csv('./T-R-test.csv')
X_test = data_test.loc[:,"T"]
y_test = data_test.loc[:,'rate']
X_test = np.array(X_test).reshape(-1, 1)
#make prediction on the training and testing data
y_train_predict = lr1.predict(X_train)
y_test_predict = lr1.predict(X_test)


r2_train = r2_score(y_train, y_train_predict)
r2_test = r2_score(y_test, y_test_predict)

print(r2_train, r2_test)

#generate new data
X_range = np.linspace(40, 90, 300).reshape(-1, 1)
y_range_predict = lr1.predict(X_range)

fig2 = plt.figure(figsize=(5,5))
plt.plot(X_range, y_range_predict)
plt.scatter(X_train, y_train)
plt.title('prediction data')
plt.xlabel('temperature')
plt.ylabel('rate')
#plt.show()

#generate new features 多项式模型
poly2 = PolynomialFeatures(degree=2)

X_2_train = poly2.fit_transform(X_train)
X_2_test = poly2.transform(X_test)

lr2 = LinearRegression()
lr2.fit(X_2_train, y_train)

y2_train_predict = lr2.predict(X_2_train)
y2_test_predict = lr2.predict(X_2_test)


r2_2_train = r2_score(y_train, y2_train_predict)
r2_2_test = r2_score(y_test, y2_test_predict)

print('r2_2',r2_2_train, r2_2_test)

#generate new data
X2_range = poly2.transform(X_range)
y2_range_predict = lr2.predict(X2_range)

fig3 = plt.figure(figsize=(5,5))
plt.plot(X_range, y2_range_predict)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.title('polynomial prediction result')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()