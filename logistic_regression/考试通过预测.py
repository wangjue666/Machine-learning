import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv('./examdata.csv')

#visualize the data
fig1 = plt.figure()
#Add label mask
mask = data.loc[:,'Pass'] == 1
passed = plt.scatter(data.loc[:,'Exam1'][mask], data.loc[:,'Exam2'][mask])
failed = plt.scatter(data.loc[:,'Exam1'][~mask], data.loc[:,'Exam2'][~mask])
plt.title("Exam1-Exam2")
plt.xlabel('Exam1')
plt.ylabel("Exam2")
plt.legend((passed, failed), ('passed','failed'))

#plt.show()

X = data.drop(['Pass'], axis = 1)
y = data.loc[:,'Pass']
X1 = data.loc[:,'Exam1']
X2 = data.loc[:,'Exam2']

#establish the model and train it 
LR = LogisticRegression()
LR.fit(X, y)

#show the predicted result and its accuracy
y_predict = LR.predict(X)

accuracy = accuracy_score(y, y_predict)
print('准确率是',accuracy)

y_test = LR.predict([[70,65]])
print(y_test)


#边界函数 = theta0 + theta1*x1 + theta2*x2 = 0

theta0 = LR.intercept_
theta1,theta2 = LR.coef_[0][0],LR.coef_[0][1]

print(theta0, theta1, theta2)

X2_new = -(theta0+theta1*X1)/theta2

fig3 = plt.figure()
passed = plt.scatter(data.loc[:,'Exam1'][mask], data.loc[:,'Exam2'][mask])
failed = plt.scatter(data.loc[:,'Exam1'][~mask], data.loc[:,'Exam2'][~mask])

plt.legend((passed, failed), ('passed','failed'))
plt.plot(X1, X2_new)
plt.title("Exam1-Exam2")
plt.xlabel('Exam1')
plt.ylabel("Exam2")
plt.show()