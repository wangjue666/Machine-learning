import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
#load the data
data = open('flare.txt').read()
#移除换行符
data = data.replace('\n', '').replace('\r', '')
data[:50]

#字符去重处理
letters = list(set(data))
num_letters = len(letters)

#建立字典
#int to char
int_to_char = {a: b for a, b in enumerate(letters)}
#char to int
char_to_int = {b: a for a, b in enumerate(letters)}


#滑动窗口提取数据
def extract_data(data, slide):
    x = []
    y = []
    for i in range(len(data) - slide):
        x.append([a for a in data[i:i + slide]])
        y.append(data[i + slide])
    return x, y


#字符到数字的批量转化
def char_to_int_Data(x, y, char_to_int):
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[char] for char in x[i]])
        y_to_int.append([char_to_int[char] for char in y[i]])
    return x_to_int, y_to_int


#实现输入字符文章的批量处理，输入整个字符，滑动窗口大小，转化字典
def data_preprocessing(data, slide, num_letters, char_to_int):
    char_Data = extract_data(data, slide)
    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())
    Input_RESHAPED = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(
        0,
        10,
        size=[Input_RESHAPED.shape[0], Input_RESHAPED.shape[1], num_letters])
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i, j, :] = to_categorical(Input_RESHAPED[i, j],num_classes=num_letters)
    return new, Output




#extract X and y from text data
time_step = 20
X, y = data_preprocessing(data, time_step, num_letters, char_to_int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=10)
y_train_category = to_categorical(y_train, num_letters)

model = Sequential()
model.add(LSTM(units=20,input_shape=(X_train.shape[1], X_train.shape[2]),activation='relu'))
model.add(Dense(units=num_letters, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


#train the model
model.fit(X_train, y_train_category, batch_size=1000, epochs=5)

#make prediction based on the training data
y_train_predict = model.predict_classes(X_train)
y_train_predict_char = [int_to_char[i] for i in y_train_predict]