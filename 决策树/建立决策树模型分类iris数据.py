import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
data = pd.read_csv("./iris.csv")

#denfine x and y
X = data.drop(['Species', 'label'], axis = 1)
y = data['label']

print(y.shape,X.shape)


#extablish the decision tree model
dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
dc_tree.fit(X, y)


#evaluate the model
y_predict = dc_tree.predict(X)

print('准确率',accuracy_score(y, y_predict))

#visualize the tree
fig = plt.figure(figsize = (10, 10))

tree.plot_tree(dc_tree,filled=True)
plt.show()