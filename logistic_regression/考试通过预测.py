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
#plt.show()

#二阶边界函数： theta0 + theta1*x1 + theta2*x2 + theta3*(x1)^2 + theta4*(x2)^2 + theta5*x1*x2 = 0

X1_2 = X1 * X1
X2_2 = X2 * X2
X1_X2 = X1 * X2

X_new = {'X1': X1, 'X2': X2, 'X1_2': X1_2, 'X2_2': X2_2, 'X1_X2': X1_X2}

X_new = pd.DataFrame(X_new)

#establish new model and train
LR2 = LogisticRegression()
LR2.fit(X_new, y)

y2_predict = LR2.predict(X_new)
accuracy2 = accuracy_score(y, y2_predict)

print(accuracy2)

theta0 = LR2.intercept_
theta1,theta2,theta3,theta4,theta5 = LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]


#根据一元二次方程 根求解公式
x1_new = X1.sort_values()
a = theta4
b = theta5*x1_new+theta2
c = theta0 + theta1*x1_new+theta3*x1_new*x1_new
x2_new_boundary = (-b+np.sqrt(b*b-4*a*c))/(2*a)


fig4 = plt.figure()
passed = plt.scatter(data['Exam1'][mask],data['Exam2'][mask])
failed = plt.scatter(data['Exam1'][~mask],data['Exam2'][~mask])
plt.plot(x1_new,x2_new_boundary)
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()