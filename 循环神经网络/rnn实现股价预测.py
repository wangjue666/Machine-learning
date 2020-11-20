import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
data = pd.read_csv('zgpa_train.csv')

price = data.loc[:, 'close']

# 归一化处理
price_norm = price / max(price)

fig1 = plt.figure(figsize=(8, 5))
plt.plot(price_norm)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
# plt.show()

# define X and y
# define method to extract X and y


def extract_data(data, time_step):
    X = []
    y = []
    for i in range(len(data) - time_step):
        X.append([a for a in data[i:i + time_step]])  #左开右闭
        y.append(data[i + time_step])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, np.array(y)


time_step = 8
X, y = extract_data(price_norm, time_step)
# set up the model

model = Sequential()
#add RNN layer
model.add(SimpleRNN(units=5, input_shape=(time_step, 1), activation='relu'))
#add ouput layer
model.add(Dense(units=1, activation='linear'))
#configure the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model.fit(X, y, batch_size=30, epochs=30)



#make prediction based on the training data
y_train_predict = model.predict(X) * max(price)
y_train = [i*max(price) for i in y]
fig2 = plt.figure(figsize=(8, 5))
plt.plot(y_train, label='real price')
plt.plot(y_train_predict, label = 'predict price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
#plt.show()


data_test = pd.read_csv('zgpa_test.csv')

price_test = data_test.loc[:, 'close']

#extract X_test and y_test
price_test_norm = price_test / max(price)
X_test_norm, y_test_norm = extract_data(price_test_norm, time_step)

#make prediction based on the test data
y_test_predict = model.predict(X_test_norm) * max(price)
y_test = [i * max(price) for i in y_test_norm]

fig3 = plt.figure(figsize=(8, 5))
plt.plot(y_test, label='real price_test')
plt.plot(y_test_predict, label = 'predict price_test')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()