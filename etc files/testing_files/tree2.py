import pandas as pd
import numpy as np



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
        self.mse = None
        self.N = None

        self.split_col = None
        # Split value for the split_col
        self.split_val = None
        # The mse for the current or best split
        self.split_mse = None

        self.left = None
        self.right = None

        self.split()

    def split(self):
        # Iterate over every column in X and try to split
        for i in range(self.X.shape[1]):
            self.find_best_split(i)



            # Once done with finding the split, actually split and # create two subtrees
        self.left = DecisionTreeRegressor()
        self.left.internal_fit()
        self.right = DecisionTreeRegressor()
        self.right.internal_fit()

    def find_best_split(self, i):



        self.split_col = i
        # self.X.sort_values(self.X.columns[i], inplace=True)
        X = self.X.values[self.indices, i]
        y = self.y.values[self.indices]

        # print("X: ", X)
        # print("selfX: ", self.X)


        X, y = zip(*sorted(zip(X,y)))


        # for i in range(len(y)):
        #     left = y[:i]
        #     right = y[i:]
        #     predValue = np.sum(y)/len(y-i)
        #     y = pow(y - predValue,2)
        #     MSE = np.sum(y)/len(y)

        #

        total_MSE_prev = 10000000000000000000000000000000000
        splitLeft = None
        splitRight = None
        X_split_value = 0
        total_MSE = 0
        value = 0
        splitPoint = 0
        for i in range(len(y)):
            if i != 1:

                left = y[:i]
                right = y[i:]
                MSE_a = 0
                if len(left) > 0:
                    valueLeft = np.sum(left)/i
                    diff_a = valueLeft - left
                    MSE_a = np.sum(pow(diff_a, 2)) / i
                ##Part 2
                valueRight = np.sum(right)/ (len(y) - i)
                ##Returns average of the dataset in the first go-around
                if i == 0:
                    value = valueRight
                    print("Total Value of Dataset:", value)

                diff_b = valueRight - right
                MSE_b = np.sum(pow(diff_b, 2)) / len(right)
                total_MSE = MSE_a + MSE_b
                print("Current MSE: ", total_MSE)
                print("Previous MSE: ", total_MSE_prev)
                if total_MSE > total_MSE_prev:
                    # print("Length of y: ", len(y))
                    print("Data is split")
                    print("Self.X:", self.X)
                    X_feature_split = self.X.iloc[:, self.split_col]
                    X_split_value = (X_feature_split[i - 2] + X_feature_split[i - 1]) / 2
                    print("Split_Value: ", X_split_value)
                    splitPoint = i - 1
                    print(i - 1)
                    splitLeft = self.X.iloc[:splitPoint]
                    splitRight = self.X.iloc[splitPoint:]
                    left_data = y[:splitPoint]
                    right_data = y[splitPoint:]
                    break
                else:
                    total_MSE_prev = total_MSE

        # for i in range(self.N):
        #     left = X <= X[i]
        #     right = X > X[i]

        # X_feature_split = X.iloc[:, i]
        # X_split_value = (X_feature_split[i - 2] + X_feature_split[i - 1]) / 2


#
# ########
# import pandas as pd
# import numpy as np
# csv_df = pd.read_csv("Income.csv")
# X = csv_df.drop('Income', axis=1)
# y = csv_df['Income']
#
# Z = DecisionTreeRegressor(0,0)
# Z.fit(X,y)
