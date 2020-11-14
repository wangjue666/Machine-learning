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