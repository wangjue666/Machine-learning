import pandas as pd
import numpy as np

#加载数据
data = pd.read_csv('custom.csv')

X = data.drop(['y'], axis= 1)

print(X)

Y = data.loc[:,'y']
print(Y)

#建立模型
from sklearn.naive_bayes import CategoricalNB

#建立模型实例
model = CategoricalNB()

#训练模型
model.fit(X, Y)

y_prdict_prob = model.predict_proba(X)

print(y_prdict_prob)

#输出预测y
y_predict = model.predict(X)
print(y_predict)

#计算模型准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y, y_predict)
print(accuracy)


#测试样本预测
X_test = np.array([[0,0,0,1,1,0]])
#模型的可能性预测
y_test_proba = model.predict_proba(X_test)
print(y_test_proba)
#模型预测结果
y_test = model.predict(X_test)
print(y_test)