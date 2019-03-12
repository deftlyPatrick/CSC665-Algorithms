import pandas as pd
import numpy as np
import metrics as met
import math
import features as fet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as RF

csv_df = pd.read_csv("Income.csv")
# X, y = ft.preprocess_ver_1(csv_df)
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

RANDOM_STATE = 10
X_train, X_test, y_train, y_test = fet.train_test_split(X, y, test_size=0.2,shuffle=True, random_state=RANDOM_STATE)


Z = RF(n_estimators=100, random_state=RANDOM_STATE)
Z.fit(X,y)
a = Z.predict(X)
print(a)

