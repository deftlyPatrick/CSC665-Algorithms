import pandas as pd
import numpy as np
import metrics as met
import features


class DecisionTreeRegressor:
    def __init__(self, max_depth, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def __repr__(self):
        raise NotImplementedError()

    def fit(self, X: pd.DataFrame, y: np.array):
        self.N = X.shape[0]
        self.indices = range(self.N)
        self.internal_fit(X, y, self.indices,0)

    def internal_fit(self, X, y, indices, depth):
        self.X = X
        self.y = y
        self.depth = depth
        self.indices = indices


        # Calculate value
        self.value = None
        self.mse = 10000000000000
        # self.N = None

        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = 10000000000000

        self.left = None
        self.right = None

        self.split()

    def split(self):
        # Iterate over every column in X and try to split


        for i in range(self.X.shape[1]):
            self.find_best_split(i)

        print("Split Column: ", X.columns[self.split_col])
        # print("Split Value: ", self.split_val)
        # print("Split MSE: ", self.split_mse)
        print("MSE: ", self.mse)
        print("Value <=", self.value)
        print("Depth: ", self.depth)
        print("Sample: ", self.N)
        self.depth += 1



        # Once done with finding the split, actually split and # create two subtrees
        self.indices_left = np.where(self.left)[0]
        if (len(self.indices_left) != int(self.min_samples_leaf-1)) and (self.depth < self.max_depth):
            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(X, y, self.indices_left, self.depth)
        else:
            return


        self.indices_right = np.where(self.right)[0]
        if len(self.indices_right) != int(self.min_samples_leaf-1) and (self.depth < self.max_depth):
            self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.right.internal_fit(X, y, self.indices_right,self.depth)
        else:
            return


    def find_best_split(self, i):
        # self.split_col = i

        X = self.X.values[self.indices, i]
        y = self.y.values[self.indices]
        self.N = len(y)
        # best_mse = 100000000000

        if self.N == 1:
            self.value = y
            self.split_col = i

        for j in range(self.N):
            left = X <= X[j]
            right = X > X[j]
            if j == 0:
                self.mse = met.mse(y.mean(),y)
            mse_left = met.mse(y[left].mean(), y[left])
            mse_right = met.mse(y[right].mean(), y[right])
            total_mse = mse_left + mse_right
            if total_mse < self.split_mse:
                self.split_mse = total_mse
                best_left = left
                best_right = right
                self.value = y.mean()
                self.left = best_left
                self.right = best_right
                self.split_col = i
                self.split_val =X[j]



#
# ########
# import pandas as pd
# import numpy as np


csv_df = pd.read_csv("TrainIncome.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

Z = DecisionTreeRegressor(5, 1)
Z.fit(X, y)
