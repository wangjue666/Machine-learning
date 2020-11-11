from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()


X = iris.data
y = iris.target

#创建实例
knn = KNeighborsClassifier(n_neighbors=1)
#模型训练
knn.fit(X, y)

r1 = knn.predict([[1,2,3,4], [2,4,1,2]])
print(r1)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X, y)
r5 = knn5.predict([[1,2,3,4], [2,4,1,2]])
print(r5)

#确认模型结构
print(knn5)


y5_pred = knn5.predict(X)

print('k=5准确率是',accuracy_score(y, y5_pred))

y_pred = knn.predict(X)
print('k=1准确率是',accuracy_score(y, y_pred))


#数据分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.4)

print('分离后数据维度', X_train.shape)

#分离数据集的训练与评估
knn_5_s = KNeighborsClassifier(n_neighbors=5)
knn_5_s.fit(X_train, y_train)

y_train_pred = knn_5_s.predict(X_train)
y_test_pred = knn_5_s.predict(X_test)

print("分离后数据预测", accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred))
