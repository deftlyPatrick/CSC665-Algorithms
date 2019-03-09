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

        self.depth += 1
        print("Split Column: ", X.columns[self.split_col])
        # print("Split Value: ", self.split_val)
        # print("Split MSE: ", self.split_mse)
        print("MSE: ", self.mse)
        print("Value: ", self.value)
        print("Depth: ", self.depth)
        print("Sample: ", self.N)



        # Once done with finding the split, actually split and # create two subtrees

        self.indices_left = np.where(self.left)[0]
        if len(self.indices_left) != 0:
            self.left = DecisionTreeRegressor(0, 0)
            self.left.internal_fit(X, y, self.indices_left, self.depth)
        else:
            return

        self.indices_right = np.where(self.right)[0]
        if len(self.indices_left) != 0:
            self.right = DecisionTreeRegressor(0,0)
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
csv_df = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Income2.csv")
X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

Z = DecisionTreeRegressor(30, 50)
Z.fit(X, y)


# total_MSE_prev = 10000000000000000000000000000000000
#         splitLeft = None
#         splitRight = None
#         X_split_value = 0
#         total_MSE = 0
#         value = 0
#         splitPoint = 0
#         for i in range(len(y)):
#             if i != 1:
#
#                 left = y[:i]
#                 right = y[i:]
#                 MSE_a = 0
#                 if len(left) > 0:
#                     valueLeft = np.sum(left)/i
#                     diff_a = valueLeft - left
#                     MSE_a = np.sum(pow(diff_a, 2)) / i
#                 ##Part 2
#                 valueRight = np.sum(right)/ (len(y) - i)
#                 ##Returns average of the dataset in the first go-around
#                 if i == 0:
#                     value = valueRight
#                     print("Total Value of Dataset:", value)
#
#                 diff_b = valueRight - right
#                 MSE_b = np.sum(pow(diff_b, 2)) / len(right)
#                 total_MSE = MSE_a + MSE_b
#                 print("Current MSE: ", total_MSE)
#                 print("Previous MSE: ", total_MSE_prev)
#                 if total_MSE > total_MSE_prev:
#                     # print("Length of y: ", len(y))
#                     print("Data is split")
#                     print("Self.X:", self.X)
#                     X_feature_split = self.X.iloc[:, self.split_col]
#                     X_split_value = (X_feature_split[i - 2] + X_feature_split[i - 1]) / 2
#                     print("Split_Value: ", X_split_value)
#                     splitPoint = i - 1
#                     # print(i - 1)
#                     splitLeft = self.X.iloc[:splitPoint]
#                     splitRight = self.X.iloc[splitPoint:]
#                     left_data = y[:splitLeft]
#                     right_data = y[splitRight:]
#
#                     ####
#                     self.split_val = X_split_value
#                     self.split_mse = total_MSE_prev
#                     self.split_col = i
#
#                     #subtrees
#                     self.left = left_data
#                     self.right = right_data
#                     break
#                 else:
#                     total_MSE_prev = total_MSE
#                     self.mse = total_MSE_prev
#
#
#         # for i in range(self.N):
#         #     left = X <= X[i]
#         #     right = X > X[i]
#
#         # X_feature_split = X.iloc[:, i]
#         # X_split_value = (X_feature_split[i - 2] + X_feature_split[i - 1]) / 2
#     #
#     # def predict(self, X: pd.DataFrame):
#     #     pass
#     #
#     # def score(self, X: pd.DataFrame, y: np.array):
#     #     return metrics.rsq(predict(X), y)
#
#
# #
# # ########
# # import pandas as pd
# # import numpy as np
