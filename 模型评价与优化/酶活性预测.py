import pandas as pd
import numpy as np

data_train = pd.read_csv('./T-R-train.csv')


#define x_train AND y_train
X_train = data_train.loc[:,"T"]
y_train = data_train.loc[:,'rate']
