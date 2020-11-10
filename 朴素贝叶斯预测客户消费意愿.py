import pandas as pd
import numpy as np

#加载数据
data = pd.read_csv('custom.csv')

print(data.head())

X = data