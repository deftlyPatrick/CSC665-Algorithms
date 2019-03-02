import numpy as np
import pandas as pd




class tester():
    def __init__(self):
        csv_df = pd.read_csv("Income.csv")
        csv_df.sort_values(['Education'], inplace=True)
        self.X = csv_df


    def testFunc(self):

        self.N = self.X.shape[0]
        self.indices = range(self.N)
        self.find_best_split(0)

    def find_best_split(self, i):
        self.split_col = i

        X = self.X.values[self.indices, i]
        print(X)

        # for i in range(self.N):
        #     print("X:", X)
        #     print("X[i]",X[i])
        #     left = X <= X[i]
        #     print("Left:", left)
        #     right = X > X[i]
        #     print("Right:", right)




        print(X)



