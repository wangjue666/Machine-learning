from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/training_set',target_size=(50,50),batch_size=32,class_mode='binary')
model = Sequential()

#卷积层
model.add(Conv2D(32,(3,3),input_shape=(50,50,3), activation='relu'))
#池化层
model.add(MaxPool2D(pool_size=(2, 2)))

#卷积层
model.add(Conv2D(32,(3,3), activation='relu'))
#池化层
model.add(MaxPool2D(pool_size=(2, 2)))

#flatten layer
model.add(Flatten())
#全连接层
model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))


#configure the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#train the model
model.fit(training_set, epochs=25)


#accuracy on the training data
accuracy_train = model.evaluate_generator(training_set)
print(accuracy_train)

#评估测试集
test_set = train_datagen.flow_from_directory('./dataset/test_set',target_size=(50,50),batch_size=32,class_mode='binary')
accuracy_test = model.evaluate_generator(test_set)
print(accuracy_test)

#加载单张图片测试

pic_dog = 'dog.jpg'
pic_dog = load_img(pic_dog,target_size=(50,50))
pic_dog = img_to_array(pic_dog)
pic_dog = pic_dog/255
pic_dog = pic_dog.reshape(1,50,50,3)
result_dog = model.predict_classes(pic_dog)
print('狗的预测结果', result_dog)

#不准确
pic_cat = 'cat.jpg'
pic_cat = load_img(pic_cat,target_size=(50,50))
pic_cat = img_to_array(pic_cat)
pic_cat = pic_cat/255
pic_cat = pic_cat.reshape(1,50,50,3)
result_cat = model.predict_classes(pic_cat)
print('猫的预测结果', result_cat)
