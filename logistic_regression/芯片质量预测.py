import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('xinpian.txt',header=None)
columns = {0:'test1',1:'test2',2:'pass'}
data.rename(columns = columns,inplace=True)

mask = data['pass'] ==1
passed = plt.scatter(data['test1'][mask],data['test2'][mask])
failed = plt.scatter(data['test1'][~mask],data['test2'][~mask])
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
#plt.show()

x = data.drop('pass',axis =1)
y = data['pass']
x1 = data['test1']
x2 = data['test2']
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2
x_new = {'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
x_new = pd.DataFrame(x_new)
LR2 = LogisticRegression()
LR2.fit(x_new,y)
x1_new = x1.sort_values()
theta0 = LR2.intercept_
theta1,theta2,theta3,theta4,theta5 = LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]
a = theta4
b = theta5*x1_new + theta2
c = theta0 + theta1*x1_new+theta3*x1_new*x1_new
x2_new_boundary = (-b+np.sqrt(b*b-4*a*c))/(2*a)
fig5 = plt.figure()
passed = plt.scatter(data['test1'][mask],data['test2'][mask])
failed = plt.scatter(data['test1'][~mask],data['test2'][~mask])
plt.plot(x1_new,x2_new_boundary)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
#plt.show()


#define f(x)
def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta0 + theta1 * x+ theta3 * x* x
    x2_new_boundary1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    x2_new_boundary2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return x2_new_boundary1,x2_new_boundary2


x2_new_boundary1 =[]
x2_new_boundary2 = []

for x in x1_new:
    item1,item2 = f(x)
    x2_new_boundary1.append(item1)
    x2_new_boundary2.append(item2)


fig6 = plt.figure()
passed = plt.scatter(data['test1'][mask],data['test2'][mask])
failed = plt.scatter(data['test1'][~mask],data['test2'][~mask])
plt.plot(x1_new,x2_new_boundary1)
plt.plot(x1_new,x2_new_boundary2)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
#plt.show()    


x1_range = [-0.9 + x/10000 for x in range(0,19000)]
x1_range = np.array(x1_range)
x2_new_boundary1 =[]
x2_new_boundary2 = []
for x in x1_range:
    x2_new_boundary1.append(f(x)[0])
    x2_new_boundary2.append(f(x)[1])
fig6 = plt.figure()
passed = plt.scatter(data['test1'][mask],data['test2'][mask])
failed = plt.scatter(data['test1'][~mask],data['test2'][~mask],marker='^')
plt.plot(x1_range,x2_new_boundary1)
plt.plot(x1_range,x2_new_boundary2)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
plt.show()
