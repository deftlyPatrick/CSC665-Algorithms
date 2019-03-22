from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import pandas as pd


csv_df = pd.read_csv("TrainIncome.csv")
# model = tree.DecisionTreeRegressor()
# csv_df = pd.read_csv("Income.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']
# model.fit(X,y)
# print(model.score(X,y))
# print(model.predict([[30,100]]))
#

Z = DecisionTreeRegressor()
Z.fit(X,y)
a = Z.predict([[30,100]])
print(a)

##Result :
# 1.0
# [53.53]
