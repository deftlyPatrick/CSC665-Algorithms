import pandas as pd
import numpy as np
<<<<<<< HEAD
import metrics
import features
#-----------------------------------------------------------------------------------------
=======
import metrics as met
import os
import features as ft


>>>>>>> 2faf0e9d061fe65a0b2752d0389be516e32bd6de
class DecisionTreeRegressor:
    def __init__(self, max_depth, min_samples_leaf):
        assert min_samples_leaf > 0

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
<<<<<<< HEAD

        # Will be assigned in fit()
        self.depth = 0
        self.left = None
        self.right = None
=======
        # self.my_data = []
    def __repr__(self):
        raise NotImplementedError()
>>>>>>> 2faf0e9d061fe65a0b2752d0389be516e32bd6de

        # self.N = None
        self.value = None
        self.mse = None
        self.indices = None

        self.X = None
        self.y = None

        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = None
    #-------------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.array):
        # self.indices = range(X.shape[0])

        self.internal_fit(
            X,
            y,
            # Indices of rows to be used in the tree
            # All rows will be used at the top level; then this array will depend
            # on the split
            np.array(range(X.shape[0])),
            # Initial depth.
            0)
    #-------------------------------------------------------------------------------------
    def internal_fit(self, X, y, indices: np.array, depth):
        self.X = X
<<<<<<< HEAD
        self.y = y
=======
        self.y = np.array(y)
        self.depth = depth
>>>>>>> 2faf0e9d061fe65a0b2752d0389be516e32bd6de
        self.indices = indices
        self.depth = depth

        # Calculate value
        self.value = y[indices].mean()
        # self.value = [12, 1, 3]
        self.mse = ((y[indices] - self.value) ** 2).mean()
        # self.N = indices.shape[0]

        # The following values will be set during training/splitting
        # Index of a column on which we split
        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = None

        # Left and right subtrees, if not leaf
        self.left = None
        self.right = None

        if self.max_depth is None or depth < self.max_depth:
            self.split(indices)
    #-------------------------------------------------------------------------------------
    def split(self, indices: np.array):
        # Iterate over every column in X and try to split
<<<<<<< HEAD
        for col_index in range(self.X.shape[1]):
            self.find_best_split(col_index, indices)

        # We may fail to find a split even if the max_depth permits, due to
        # min_samples_leaf. In this case we create no branches.
        if self.split_mse is not None:
            # print("Best split: ", self.depth, self.split_col, self.split_val, self.split_mse)

            # Once done with finding the split, actually split and
            # create two subtrees
            X_col = self.X.values[indices, self.split_col]

            # Left indices
            left_indices_bool = X_col <= self.split_val
            left_indices = indices[left_indices_bool]
            # print("left", left_indices)
            assert isinstance(left_indices, np.ndarray)

            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(self.X, self.y, left_indices, self.depth + 1)

            # Right indices
            right_indices_bool = X_col > self.split_val
            right_indices = indices[right_indices_bool]
            assert isinstance(right_indices, np.ndarray)

            self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.right.internal_fit(self.X, self.y, right_indices, self.depth + 1)
    #-------------------------------------------------------------------------------------
    def find_best_split(self, col_index, indices):
        X_col = self.X.values[indices, col_index]
        y = self.y[indices]

        for row_index in range(indices.shape[0]):
            left = X_col <= X_col[row_index]
            right = X_col > X_col[row_index]

            assert isinstance(left, np.ndarray)
            assert isinstance(right, np.ndarray)

            # Calculate MSE values and decide if this the best split
            # so far. If yes, set the object values: self.split_col,
            # self.split_val, etc.

            # If one of the branches has NO samples, then this is an invalid split. Skip.
            if left.any() and right.any():
                cur_mse = self.calc_mse(y[left]) + self.calc_mse(y[right])
                # print(X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum())
                if self.split_mse is None or cur_mse < self.split_mse:
                    if left.sum() >= self.min_samples_leaf and right.sum() >= self.min_samples_leaf:
                        # best split
                        self.split_mse = cur_mse
                        self.split_col = col_index
                        self.split_val = np.mean([np.max(X_col[left]), np.min(X_col[right])])
                        # print("New Best: ", X_col[row_index], cur_mse, self.calc_mse(y[left]) / left.sum(),
                        #       X_col[left].max(), X_col[right].min())
    #-------------------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame):
        result = []
        for row_index in range(X.shape[0]):
            result.append(self.internal_predict(X.values[row_index]))
        return np.array(result)
    #-------------------------------------------------------------------------------------
    def internal_predict(self, X_row: np.array):
        if self._is_leaf():
            # self.value = [12, 1, 2]
            # np.argmax(self.value) -> 0
            return self.value
            # / np.sum(self.value) -> 12 / 15 , 0.80
        else:
            if X_row[self.split_col] <= self.split_val:
                next_branch = self.left
            else:
                next_branch = self.right
            return next_branch.internal_predict(X_row)
    #-------------------------------------------------------------------------------------
    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.rsq(self.predict(X), y)
    #-------------------------------------------------------------------------------------
    def _is_leaf(self):
        assert self.left is None and self.right is None \
            or self.left is not None and self.right is not None
        return self.left is None
    #-------------------------------------------------------------------------------------
    # @staticmethod
    # noinspection PyMethodMayBeStatic
    def calc_gini(self, y):
        # value = y.mean()
        # return ((y - value) ** 2).sum()
        return np.var(y) * y.shape[0]
    #-------------------------------------------------------------------------------------
    def __repr__(self):
        # The number of tabs equal to the level * 4, for formatting
        # Print the tree for debugging purposes, e.g.
        # 0: [value, mse, samples, split_col <= split_val, if any]
        #     1: [value, mse, samples, split_col <= split_val, if any]
        #     1: [value, mse, samples, split_col > split_val, if any]
        #           2: 1: [value, mse, samples, split_col <= split_val, if any]
        # etc..
        # The number of tabs equal to the level * 4, for formatting

        tabs = "".join([" " for _ in range(self.depth * 4)])

        if not self._is_leaf():
            attr_name = self.X.columns[self.split_col]
            split_expression = attr_name + " <= " + "{}".format(self.split_val)
        else:
            split_expression = ''

        self_repr = "{}:{}[{}, val: {:0.2f}, mse: {:0.2f}, samples: {:0.2f}]\n".format(
            self.depth,
            tabs,
            split_expression,
            self.value,
            self.mse,
            self.indices.shape[0]
        )

        if self.left:
            self_repr += self.left.__repr__()
            self_repr += self.right.__repr__()

        return self_repr

csv_df = pd.read_csv("TrainIncome.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

X_train, X_test, y_train, y_test = features.train_test_split(X, y, test_size=0.10, shuffle=True, random_state=None)
Z = DecisionTreeRegressor(30, 1)
Z.fit(X, y)

a = Z.predict(X_test)
print(a)
=======

        for i in range(self.X.shape[1]):
            self.find_best_split(i)

        # print("Split Column: ", X.columns[self.split_col])
        # print("Split Value <= ", self.split_val)
        # # print("Split MSE: ", self.split_mse)
        # print("MSE: ", self.mse)
        # print("Value: ",self.value)
        # print("Depth: ", self.depth)
        # print("Sample: ", self.N)
        self.depth += 1

        if self.N == 1:
            file_bytes = int(os.path.getsize('predict.txt'))
            if file_bytes == 0:
                f = open('predict.txt', 'a')
                print(self.value, file=f)
                f.close()
            else:
                f = open('predict.txt', 'r')
                f.seek(0)
                f.close()

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

        # self.indices_left = np.where(self.left)[0]
        # self.indices_right = np.where(self.right)[0]

        # if self.depth == self.max_depth or len(self.indices_left) != int(self.min_samples_leaf) or len(self.indices_right) < int(self.min_samples_leaf):
        #     return
        # self.my_data = [1,2,3]
        self.indices_left = np.where(self.left)[0]
        if (len(self.indices_left) != int(self.min_samples_leaf - 1)) and (self.depth < self.max_depth):
            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            # print("\n")
            # print("Left: ")
            self.left.internal_fit(self.X, self.y, self.indices_left, self.depth)
            # self.internal_fit(self.X, self.y, self.indices_left, self.depth)
        else:
            return

        self.indices_right = np.where(self.right)[0]
        if (len(self.indices_right) != int(self.min_samples_leaf - 1)) and (self.depth < self.max_depth):
            self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            # print("\n")
            # print("Right: ")
            self.right.internal_fit(self.X, self.y, self.indices_right,self.depth)
            # self.internal_fit(self.X, self.y, self.indices_right, self.depth)
        else:
            return

    def find_best_split(self, i):
        # self.split_col = i

        X = self.X.values[self.indices, i]

        y = self.y[self.indices]
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

                target_num = X[j]
                temp_a = np.sort(X)
                found_indx = np.where(temp_a==target_num)[0]

                self.split_val = (temp_a[found_indx[-1]+1] + temp_a[found_indx[-1]])/2

                # self.split_val = X[j]


    def predict(self, X: pd.DataFrame):
        y_pred_list = np.loadtxt('predict.txt')
        return y_pred_list





    def score(self, X: pd.DataFrame, y: np.array):
         return met.rsq(self.predict(X), y)





# ######################################################################################################################
# #
# csv_df = pd.read_csv("TrainIncome.csv")
# X = csv_df.drop('Income', axis=1)
# y = csv_df['Income']
# q = np.array(y)
#
# Z = DecisionTreeRegressor(30, 1)
# Z.fit(X, y)
# print(Z.predict([[30,100]]))
# print(Z.score(X,y))
>>>>>>> 2faf0e9d061fe65a0b2752d0389be516e32bd6de
