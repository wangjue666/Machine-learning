import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score
data = pd.read_csv('zhijian.csv')

#define x and y

x = data.drop(['y'],axis=1)
y = data['y']

#visualize the data
fig1 = plt.figure(figsize=(5,5))
plt.scatter(x['x1'][y==1],x['x2'][y==1],label = 'passed')
plt.scatter(x['x1'][y==0],x['x2'][y==0],label = 'failed')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
#plt.show()


#split the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=10)

#set up the model

mlp = Sequential()
mlp.add(Dense(units=20,input_dim=2,activation='sigmoid'))
mlp.add(Dense(units=1,activation='sigmoid'))
mlp.summary()

#complie the model

mlp.compile(optimizer='adam',loss='binary_crossentropy')
mlp.fit(x_train,y_train,epochs=3000)


#make prediction and calculate the accuracy
y_train_predict = mlp.predict_classes(x_train)
accuracy_train = accuracy_score(y_train,y_train_predict)

y_test_predict = mlp.predict_classes(x_test)
accuracy_test = accuracy_score(y_test,y_test_predict)

y_train_predict_from = pd.Series(i[0] for i in y_train_predict)


#generate new data for plot

xx,yy = np.meshgrid(np.arange(0,10,0.1),np.arange(0,10,0.1))
x_range = np.c_[xx.ravel(),yy.ravel()]
y_range_predict = mlp.predict_classes(x_range)
y_range_predict_form = pd.Series(i[0] for i in y_range_predict)


fig2 = plt.figure(figsize=(5,5))
plt.scatter(x['x1'][y==1],x['x2'][y==1],label = 'passed')
plt.scatter(x_range[:,0][y_range_predict_form==1],x_range[:,1][y_range_predict_form==1],label = 'passed_predict')
plt.scatter(x_range[:,0][y_range_predict_form==0],x_range[:,1][y_range_predict_form==0],label = 'failed_predict')
plt.scatter(x['x1'][y==0],x['x2'][y==0],label = 'failed')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('prediction result')
plt.show()
