from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
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