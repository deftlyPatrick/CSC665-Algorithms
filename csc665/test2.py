from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import pandas as pd


csv_df = pd.read_csv("Income.csv")
model = tree.DecisionTreeRegressor()
csv_df = pd.read_csv("Income.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']
model.fit(X,y)
print(model.score(X,y))
print(model.predict([[15,23]]))


##Result :
# 1.0
# [53.53]
