import pandas as pd
import numpy as np
# import metrics as met
# from sklearn.tree import DecisionTreeRegressor
#
#
# csv_df = pd.read_csv("Income.csv")
# X = csv_df.drop('Income', axis=1)
# y = csv_df['Income']
#
# # for i in range(len(y)):
# #     mse_pt1 = met.mse(y[:i].mean(),y[:i])
# #     mse_pt2 = met.mse(y[i:].mean(),y[i:])
# #     print("mse 1: ", mse_pt1)
# #     print("mse 2: ", mse_pt2)
#
# N = X.shape[0]
# indices = range(N)
#
# # q = X.values[indices,0]
# # print(q)
#
#
#
# def test(X: pd.DataFrame, y: np.array, indices):
#     # print("indices:", indices)
#     X_val = X.values[indices, 0]
#     y_val = y.values[indices]
#     # print(y.values[indices])
#
#
#
#     for i in range(N):
#         # print("q: ",q)
#         # print("qLeft: ", q[i])
#
#         left = X_val <= X_val[i]
#         print("left: ", left)
#         right = X_val > X_val[i]
#         print("right: ", right)
#         mse_left = met.mse(y_val[left].mean(), y_val[left])
#         mse_right = met.mse(y[right].mean(), y[right])
#         total_mse = mse_left + mse_right
#         print(total_mse)
#
#
#
#     return left,right
#
#
# t = test(X,y, indices)
#
#
#
#
# # z = y[indices]
# # print(y)
#
# # print(y[:2].mean())
# # Z = met.mse(y[2:].mean(),y[2:])
# # print(Z)

# primes = [2, 3, 5, 7]
a = np.array([2, 3, 5, 7])
for i in range(len(a)):
    b = a[i] + a[i+1]
    print(b)
