import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('anomaly_data.csv')
print(data.head())

plt.scatter(data.loc[:,'x1'],data.loc[:, 'x2'])
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')

#plt.show()

x1 = data.loc[:, 'x1']
x2 = data.loc[:, 'x2']

fig2 = plt.figure(figsize=(20,20))
plt.subplot(121)
plt.hist(x1, bins = 100)
plt.title('x1 distribution')
plt.xlabel('x1')
plt.ylabel('counts')

plt.subplot(122)
plt.hist(x2, bins = 100)
plt.title('x2 distribution')
plt.xlabel('x2')
plt.ylabel('counts')
plt.show()


#Gaussian distribution

x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()

x1_range = np.linspace(4,25,300)
x1_normal = norm.pdf(x1_range,x1_mean,x1_sigma)
x2_range = np.linspace(2,25,300)
x2_normal = norm.pdf(x2_range,x2_mean,x2_sigma)
fig3 = plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(x1_range,x1_normal)
plt.title('normal p(x1)')
plt.subplot(122)
plt.plot(x2_range,x2_normal)
plt.title('normal p(x2)')
plt.show()


#establish the model and predict

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(data)
print("权重", ad_model)
y_predict = ad_model.predict(data)
fig5 = plt.figure(figsize=(10,5))
original_data = plt.scatter(data['x1'],data['x2'],marker='x')
anomaly_data = plt.scatter(data['x1'][y_predict==-1],data['x2'][y_predict==-1],marker='o',facecolor='none',edgecolor='r',s=150)
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((original_data,anomaly_data),('original_data','anomaly_data'))
plt.show()