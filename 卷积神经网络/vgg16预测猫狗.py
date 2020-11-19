import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt


#load the data
img_path = 'cat.jpg'
img = load_img(img_path,target_size=(224,224))
img = img_to_array(img)
model_vgg = VGG16(weights='imagenet',include_top=False)
x = np.expand_dims(img,axis=0)
x = preprocess_input(x)
print(x.shape)

#轮廓特征提取
features = model_vgg.predict(x)
print(features.shape)

#flatten
features = features.reshape(1, 7*7*512)

#visiualize the data
fig = plt.figure(figsize=(5,5))
img = load_img(img_path,target_size=(224,224))
plt.imshow(img)