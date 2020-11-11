import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

path = './pima_data.csv'

pima = pd.read_csv(path)

feature_names = ['Pregnancies', 'Insulin','BMI', 'Age']

X = pima[feature_names]
y = pima.Outcome

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

#测试数据集结果预测
y_pred = logreg.predict(X_test)


#使用准确率进行评估
print(accuracy_score(y_test, y_pred))

#确认正负样本数据量
print(y_test.value_counts())

#空准确率
print('空准确率是', max(y_test.mean(), 1-y_test.mean()))

#展示部分实际结果与预测结果(25组)
print('true', y_test.values[0:25])
print('pred', y_pred[0:25])

#4个混淆矩阵因子赋值
confusion = confusion_matrix(y_test, y_pred)
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
TP = confusion[1,1]

print(TN, FP, FN, TP)


accuracy = (TP+TN)/(TP+TN+FP+FN)

print('准确率是', accuracy, accuracy_score(y_test, y_pred))

mis_rate = (FP+FN)/(TP+TN+FP+FN)

print('错误率是', mis_rate, 1-accuracy)

recall = TP / (TP+FN)
print('召回率是', recall)

specificity = TN/(TN+FP)
print('特异度是', specificity)

precision = TP/(TP+FP)
print('精确率', precision)

#F1分数 综合准确率和召回率的一个判断指标
f1_score = 2*precision * recall / (precision + recall)

print("f1分数", f1_score)