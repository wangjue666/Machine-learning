import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv("./zhijian.csv")

#define x and y
X = data.drop(['y'], axis = 1)
y = data.loc[:, 'y']

#visualize the data
fig1 = plt.figure(figsize=(5,5))
bad = plt.scatter(X['x1'][y==0],X['x2'][y==0],label = 'bad')
good = plt.scatter(X['x1'][y==1],X['x2'][y==1],label='good')
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()


#anomay dection
ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(X[y==0])
y_predict_bad = ad_model.predict(X[y==0])
print(y_predict_bad)

plt.scatter(X['x1'][y==0][y_predict_bad==-1],X['x2'][y==0][y_predict_bad==-1], marker='x', s=150)
#plt.show()

data = pd.read_csv('./zhijian_processed.csv')
X = data.drop(['y'], axis = 1)
y = data.loc[:, 'y']
print(X.shape)
#pca降维
X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_norm)
var_ratio = pca.explained_variance_ratio_
print('主成分的标准差是', var_ratio)
#fig4 = plt.figure(figsize=(5,5))
#plt.bar([1,2], var_ratio)
#plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4, test_size=0.4)

#knn model 
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_10.fit(X_train, y_train)
y_train_predict = knn_10.predict(X_train)
y_test_predict = knn_10.predict(X_test)

#calculate accuracy
accuracy_train = accuracy_score(y_train, y_train_predict)
accuracy_test = accuracy_score(y_test, y_test_predict)
print('training accuracy', accuracy_train)
print('training test', accuracy_test)

#visiualize the result and boundary
xx, yy = np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range = np.c_[xx.ravel(), yy.ravel()]


y_range_predict = knn_10.predict(x_range)
fig5 = plt.figure(figsize=(5,5))
knn_bad = plt.scatter(x_range[:,0][y_range_predict==0], x_range[:,1][y_range_predict==0])
knn_good = plt.scatter(x_range[:,0][y_range_predict==1], x_range[:,1][y_range_predict==1])
bad = plt.scatter(X['x1'][y==0],X['x2'][y==0],label = 'bad',marker='x')
good = plt.scatter(X['x1'][y==1],X['x2'][y==1],label='good')
#plt.show()

#计算混淆矩阵
cm = confusion_matrix(y_test,y_test_predict)
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
accuracy = (TP + TN)/(TP+TN+FP+FN)
recall = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
f1_score = (2*recall*precision)/(precision+recall)
print(TP, TN, FP, FN, accuracy, recall, specificity, precision, f1_score)