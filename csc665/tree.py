import pandas as pd
import numpy as np
import metrics as met
import features as ft


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
        print("Split Value <= ", self.split_val)
        # print("Split MSE: ", self.split_mse)
        print("MSE: ", self.mse)
        print("Value: ",self.value)
        print("Depth: ", self.depth)
        print("Sample: ", self.N)
        self.depth += 1



        # Once done with finding the split, actually split and # create two subtrees

        # if (len(self_indices_right) != int(self.min_samples_leaf - 1)) and (self.depth < self.max_depth):
        #     self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
        #     print("\n")
        #     print("Right: ")
        # self.right.internal_fit(X, y, self_indices_right, self.depth)
        # else:
        #     return


        # self.indices_left = np.where(self.left)[0]
        # print("self.indices_left: ", self.indices_left)
        # self.indices_right = np.where(self.right)[0]
        # print("self.indices_right: ", self.indices_right)

        self.indices_left = np.where(self.left)[0]
        self.indices_right = np.where(self.right)[0]

        if self.depth == self.max_depth or len(self.indices_left) < int(self.min_samples_leaf) or len(self.indices_right) < int(self.min_samples_leaf):
            return

        self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
        print("\n")
        print("Left: ")
        self.left.internal_fit(X, y, self.indices_left, self.depth)

        self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
        print("\n")
        print("Right: ")
        self.right.internal_fit(X, y, self.indices_right,self.depth)

    def find_best_split(self, i):
        # self.split_col = i

        X = self.X.values[self.indices, i]
        y = self.y.values[self.indices]
        self.N = len(y)

        # best_mse = 100000000000

        if self.N == 1:
            self.value = y[0]
            self.split_col = i
            self.split_val = 0

        for j in range(self.N):
            left = X <= X[j]
            right = X > X[j]
            if j == 0:
                self.mse = met.mse(y.mean(), y)

            mse_left = met.mse(y[left].mean(), y[left])
            mse_right = met.mse(y[right].mean(), y[right])
            total_mse = mse_left + mse_right
            if total_mse < self.split_mse:
                self.split_mse = total_mse
                best_left = np.full((1, len(self.y)), False, dtype=bool)
                best_left = best_left[0]
                best_right = np.full((1, len(self.y)), False, dtype=bool)
                best_right = best_right[0]
                self.indices = np.array(self.indices)
                for h in range(len(left)):
                    best_left[self.indices[h]] = left[h]
                    best_right[self.indices[h]] = right[h]
                #  best_left = left
                #  best_right = right
                self.value = y.mean()
                self.left = best_left
                self.right = best_right
                self.split_col = i
                self.split_val = X[j]
                # print("X:", X)
                # print("X[j-1]: ", X[j-1])
                # print("X[j]: ", X[j])
                # print("X[j+1]: ", X[j])
                # print("self.N: ", self.N)
                # if self.N == 2:
                #     j = 0
                #     self.split_val = (X[j] + X[j-1])/2
                # elif X[j+1]is None:
                #     self.split_val = (X[j-1] + X[j])/2
                # elif X[j-1] is None:
                #     self.split_val = (X[j] + X[j+1])/2
                # else:
                #     self.split_val = (X[j] + X[j+1])/2

    # def predict(self, X: pd.DataFrame):
    #
    #
    #
    #
    # def score(self, X: pd.DataFrame, y: np.array):
    #      return met.rsq(predict(X), y)



########################################################################################################################

csv_df = pd.read_csv("TrainIncome.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

Z = DecisionTreeRegressor(30, 1)
Z.fit(X, y)
