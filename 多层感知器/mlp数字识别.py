from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.metrics import accuracy_score


(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(type(X_train), X_train.shape)

#visualize the data
img1 = X_train[0]

fig1 = plt.figure(figsize=(3,3))

plt.imshow(img1)
plt.title(y_train[0])
#plt.show()

feature_size = img1.shape[0] * img1.shape[1]

X_train_format = X_train.reshape(X_train.shape[0],feature_size)
X_test_format = X_test.reshape(X_test.shape[0],feature_size)

#normalize the input data
x_train_normal = X_train_format/255
x_test_normal = X_test_format/255

#format output data
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)

#set up the model
mlp = Sequential()
mlp.add(Dense(units=392,activation='sigmoid',input_dim=feature_size))
mlp.add(Dense(units=392,activation='sigmoid'))
mlp.add(Dense(units=10,activation='softmax'))
mlp.summary()

#configure the model
mlp.compile(loss='categorical_crossentropy',optimizer='adam')
#train the model
mlp.fit(x_train_normal,y_train_format,epochs=10)


#evaluate the model
y_train_predict = mlp.predict_classes(x_train_normal)
accuracy_train = accuracy_score(y_train,y_train_predict)
y_test_predict = mlp.predict_classes(x_test_normal)
accuracy_test = accuracy_score(y_test,y_test_predict)
print('accuracy_train 准确度', accuracy_train,'accuracy_test准确度', accuracy_test)

img2 = X_test[10]
fig2 = plt.figure(figsize=(5,5))
plt.imshow(img2)
plt.title(y_test_predict[10])
plt.show()