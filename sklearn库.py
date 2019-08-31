import numpy as  np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

points=np.genfromtxt("./data.csv",delimiter=',')
x=points[:,0]
y=points[:,1]

def compute_cost(w,b,points):
    total_cost=0
    M=len(points)
    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        total_cost+=(y-w*x-b)**2
    return total_cost/M    


x_new=x.reshape(-1,1)
y_new=y.reshape(-1,1)
lr.fit(x_new,y_new)

#从训练好的模型中提取系数和截距
w=lr.coef_[0][0]
b=lr.intercept_[0]
cost=compute_cost(w,b,points)
print(w,b,cost);
# 用plt画散点图
plt.scatter(x,y)
#针对每一个x,计算预测的y值
pred_y=w*x+b

plt.plot(x,pred_y,c="r")

plt.show()