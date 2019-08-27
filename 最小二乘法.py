import numpy as  np
import matplotlib.pyplot as plt

points=np.genfromtxt("./data.csv",delimiter=',')

# 提取二维数组中的 X Y
x=points[:,0]
y=points[:,1]

# 用plt画散点图
plt.scatter(x,y)
plt.show()


#损失函数
def compute_cost(w,b,points):
    total_cost=0
    M=len(points)
    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        total_cost+=(y-w*x-b)**2
    return total_cost/M    
