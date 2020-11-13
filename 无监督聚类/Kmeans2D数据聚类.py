import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import estimate_bandwidth,MeanShift



data = pd.read_csv('2dData.csv')
x = data.drop('label',axis=1)
y = data['label']
fig1 = plt.figure()
label1 = plt.scatter(x['x'][y==1],x['y'][y==1])
label2 = plt.scatter(x['x'][y==2],x['y'][y==2])
label3 = plt.scatter(x['x'][y==3],x['y'][y==3])
label4 = plt.scatter(x['x'][y==4],x['y'][y==4])
label5 = plt.scatter(x['x'][y==5],x['y'][y==5])
plt.title('Unlabeled Data',fontsize = 15)
plt.xlabel('x')
plt.ylabel('y')
plt.legend((label1,label2,label3,label4,label5),('label1','label2','label3','label4','label5'))
#plt.show()

KM = KMeans(n_clusters=5,random_state=0)
KM.fit(x)
centers = KM.cluster_centers_
print(centers)
fig2 = plt.figure()
plt.scatter(centers[:,0],centers[:,1])
label1 = plt.scatter(x['x'][y==1],x['y'][y==1])
label2 = plt.scatter(x['x'][y==2],x['y'][y==2])
label3 = plt.scatter(x['x'][y==3],x['y'][y==3])
label4 = plt.scatter(x['x'][y==4],x['y'][y==4])
label5 = plt.scatter(x['x'][y==5],x['y'][y==5])
plt.legend((label1,label2,label3,label4,label5),('label1','label2','label3','label4','label5'))
#plt.show()

#test

y_predict_test = KM.predict([[5,0]])
y_predict = KM.predict(x)
fig3 = plt.figure()
plt.subplot(121)
label1 = plt.scatter(x['x'][y_predict==1],x['y'][y_predict==1])
label2 = plt.scatter(x['x'][y_predict==2],x['y'][y_predict==2])
label3 = plt.scatter(x['x'][y_predict==3],x['y'][y_predict==3])
label4 = plt.scatter(x['x'][y_predict==4],x['y'][y_predict==4])
label5 = plt.scatter(x['x'][y_predict==0],x['y'][y_predict==0])
plt.legend((label1,label2,label3,label4,label5),('label1','label2','label3','label4','label5'))
plt.subplot(122)
label1 = plt.scatter(x['x'][y==1],x['y'][y==1])
label2 = plt.scatter(x['x'][y==2],x['y'][y==2])
label3 = plt.scatter(x['x'][y==3],x['y'][y==3])
label4 = plt.scatter(x['x'][y==4],x['y'][y==4])
label5 = plt.scatter(x['x'][y==5],x['y'][y==5])
plt.legend((label1,label2,label3,label4,label5),('label1','label2','label3','label4','label5'))
#plt.show()


# correct the results
y_corrected = []
for i in y_predict:
    if i==1:
        y_corrected.append(4)
    elif i==2:
        y_corrected.append(1)
    elif i==3:
        y_corrected.append(2)
    elif i==4:
        y_corrected.append(5)
    else:
        y_corrected.append(3)


#establish a KNN model
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(x,y)
KNN.predict([[5,0]])
y_predict2 = KNN.predict(x)       

print("KNN accuracy", accuracy_score(y, y_predict2))
print(pd.value_counts(y_predict2))


#establish a meanshift model

bw = estimate_bandwidth(x,n_samples=120)
print(bw)
ms = MeanShift(bandwidth=bw)
ms.fit(x)
y_predict_ms = ms.predict(x)
print("meanshift 归类结果", pd.value_counts(y_predict_ms))
fig5 = plt.figure()
label1 = plt.scatter(x['x'][y_predict_ms==1],x['y'][y_predict_ms==1])
label2 = plt.scatter(x['x'][y_predict_ms==0],x['y'][y_predict_ms==0])
plt.legend((label1,label2),('label1','label2'))
plt.show()