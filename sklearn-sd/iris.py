from sklearn import datasets
iris = datasets.load_iris()

print(iris.data)
print(iris.feature_names)

print(iris.target)


#结果的含义
print(iris.target_names)


#确认数据类型
print(type(iris.data))


X = iris.data
y = iris.target


print(X.shape, y.shape)