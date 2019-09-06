import numpy as np
import pandas as pd

#引入数据
from sklearn.datasets import load_iris
#拆分数据为训练集和测试集
from sklearn.model_selection import train_test_split
#用来计算分类预测的准确率
from sklearn.metrics import accuracy_score


#数据预处理
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df["class"]=iris.target
df["class"]=df["class"].map({0:iris.target_names[0],1:iris.target_names[1],2:iris.target_names[2]})

df.describe()

x=iris.data
y=iris.target.reshape(-1,1)

#划分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=35,stratify=y)

print(x_train)