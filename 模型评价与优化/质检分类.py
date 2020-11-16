import pandas as pd
import numpy as py
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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
fig4 = plt.figure(figsize=(5,5))
plt.bar([1,2], var_ratio)
#plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4, test_size=0.4)