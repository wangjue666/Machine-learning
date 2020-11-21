#load the data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

import joblib

import os
data = pd.read_csv('transfer_data.csv')
data.head()


#difine X and y
X = np.array(data['x']).reshape(-1, 1)
y = data['y']

#visualize the data
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig1 = plt.figure()
plt.scatter(X, y, label='data1')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()


# model1 = Sequential()
# model1.add(Dense(units=50, input_dim=1, activation='relu'))
# model1.add(Dense(units=50, activation='relu'))
# model1.add(Dense(units=1, activation='linear'))
# model1.compile(optimizer='adam', loss='mean_squared_error')
# model1.summary()

#train the model
#model1.fit(X, y, epochs=200)


#make prediction
#y_predict = model1.predict(X)
fig2 = plt.figure(figsize=(7,5))
# plt.scatter(X, y, label='data1')
# plt.plot(X, y_predict, 'g', label='predict1 epochs=200')
# plt.legend()
#plt.show()

#save the model
#joblib.dump(model1, 'model1.m')


#加载已生成的模型
model2 = joblib.load('model1.m')
data2 = pd.read_csv('transfer_data2.csv')
X2 = np.array(data2['x2']).reshape(-1, 1)
y2 = data2['y2']
y2_predict = model2.predict(X2)
fig3 = plt.figure(figsize=(7,5))
plt.scatter(X, y, label='data1')
plt.scatter(X2, y2, label='data2')
plt.plot(X2, y2_predict, 'g', label='predict2 epochs=200')
plt.legend()
plt.show()