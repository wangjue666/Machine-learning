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

##核心算法实现

#距离函数
def l1_distance(a,b):
    return np.sum(np.abs(a-b),axis=1)
def l2_distance(a,b):
    return np.sqrt(np.sum( (a-b)**2 ,axis=1))

#分类器实现
class KNN(object):
    #定义一个初始化方法,__init__的构造方法
    def __init__(self,n_neighbors=1,dist_func=l1_distance):
        self.n_neighbors=n_neighbors
        self.dist_func=dist_func
    #训练模型的方法
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    #模型的预测方法
    def predict(self,x):
     
        #初始化预测分类数组
        y_pred=np.zeros( (x.shape[0],1), dtype=self.y_train.dtype)
      
        #遍历输入的x的数据点
        for i,x_test in enumerate(x):
            #x_test跟所有训练数据计算距离
            distances=self.dist_func(self.x_train,x_test)    
            #得到的距离做递增排序
            nn_index=np.argsort(distances)
            #选取最小的K个点，保存他们对应的分类类别
            nn_y=self.y_train[ nn_index[:self.n_neighbors]].ravel()
            #统计类别出现频率最高的那个 赋给y_pred[i]
            y_pred[i]=np.argmax(np.bincount(nn_y))
        return y_pred
        

# TEST        
knn=KNN()

knn.fit(x_train,y_train)

result_list=[]

for p in [1,2]:
    knn.dist_func=l1_distance if p==1 else l2_distance
    for k in range (1,10,2):
        knn.n_neighbors=k
        y_pred=knn.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        result_list.append([k,"l1_distance" if p==1 else "l2_distance",accuracy])

df=pd.DataFrame(result_list,columns=["k","距离函数","预测准确率"])

print(df)