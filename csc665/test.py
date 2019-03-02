import pandas as pd
import numpy as np


import tree
# csv_df = pd.read_csv("Income.csv")
# csv_df.sort_values(['Education'], inplace=True)
# # X = csv_df.drop('Income', axis=1)
# X = csv_df
# y = csv_df['Income']

# #multiple 5 to ONLY EDUCATION
# X.Education *= 5


#multiple 500 to ENTIRE dataframe
# a = X * 500
# print(a)



# z = tree.DecisionTreeRegressor(0,0)
# z.fit(X,y)



# csv_df = pd.read_csv("Income.csv")
# csv_df.sort_values(['Education'], inplace=True)
# X = csv_df.drop('Income', axis=1)
# y = csv_df['Income']


# csv_df = pd.read_csv("Income.csv")
# y = csv_df['Seniority']
# X = csv_df.drop('Income', axis=1)
# print(X)
# print(y)
# left = csv_df <= csv_df[1]
# csv_df.sort_values(['Education'], inplace=True)
# print(csv_df.head())




# csv_df = pd.read_csv("Income.csv")
# csv_df.sort_values(['Education'], inplace=True)
# X = csv_df.drop('Income', axis=1)
# y = csv_df['Income']
#
# print(X)
# print(y)
#
# y = np.array([ 18.57
#  ,21.39
#  ,22.64
#  ,17.61
#  ,53.53
#  ,74.61
#  ,72.08
#  ,90.81
#  ,78.81 ])
def splitFunc(X, target, colm):
    # X.sort_values(['Education'], inplace=True)
    df = X
    df.sort_values(X.columns[colm], inplace=True)
    # X = csv_df.drop('Income', axis=1)
    df.reset_index(drop=True, inplace=True)
    y = df[target]

    total_MSE_prev = 10000000000000000000000000000000000
    depth = 0
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
                valueLeft = np.sum(left/i)
                diff_a = valueLeft - left
                MSE_a = np.sum(pow(diff_a, 2)) / i
            ##Part 2
            valueRight = np.sum(right/(len(y)-i))

            ##Returns average of the dataset in the first go-around
            if i == 0:
                value = valueRight
                print("Total Value of Dataset:", value)

            diff_b = valueRight-right
            MSE_b = np.sum(pow(diff_b,2))/len(right)
            total_MSE = MSE_a + MSE_b
            print("Current MSE: ",total_MSE)
            print("Previous MSE: ", total_MSE_prev)
            if total_MSE > total_MSE_prev:
                # print("Length of y: ", len(y))
                print("Data is split")
                xtemp = df.iloc[:,colm]
                X_split_value = (xtemp[i-2] + xtemp[i-1])/2
                print("Split_Value: ",X_split_value)
                splitPoint = i-1
                print(i-1)
                splitLeft = df.iloc[:splitPoint]
                splitRight = df.iloc[splitPoint:]
                depth+=1
                break
            else:
                total_MSE_prev = total_MSE

            # print("Total MSE pt2:", total_MSE)
            # print("first part: ",left)
            # print("MSE_b :", MSE_b)

    return_val = {'splitPoint':splitPoint, 'MSE': total_MSE, 'Value': value, 'Split Value': X_split_value, 'Left Tree': splitLeft, 'Right Tree': splitRight}
    return return_val

csv_df = pd.read_csv("Income.csv")


for i in range(csv_df.shape[1]):
    Z = splitFunc(csv_df, 'Income', i)
    print(i)


# Z = splitFunc(csv_df,'Income', 0)
# # Z = splitFunc(csv_df,'Income', 1)
#
# csv2_df = Z['Right Tree']
# Z = splitFunc(csv2_df,'Income', 0)


# Z = splitFunc(csv_df,'Income', 1)
# print(splitLeft)
# print(splitRight)



# splitLeft.sort_values(['Education'], inplace=True)
# splitLeft.reset_index(drop=True, inplace=True)
# # X = csv_df.drop('Income', axis=1)
# X = splitLeft
# y = splitLeft['Income']

# Z = splitFunc(y)
# q = Z['splitPoint']
# splitLeft = splitLeft.iloc[:q]
# splitRight = splitLeft.iloc[q:]
#
# print(splitLeft)
# print(splitRight)
# ################################################################################################################################################
# ind_left = range(0,splitPoint)
# ind_right = range(splitPoint, len(y))
#
# print(ind_left)
# print(ind_right)
#
# splitLeft = csv_df.iloc[:splitPoint]
# splitRight = csv_df.iloc[splitPoint:]
# print(splitLeft)
# print(splitRight)
#
# splitLeft.sort_values(['Education'], inplace=True)
# X = splitLeft
# y = splitLeft['Income']
#
#
# total_MSE_prev = 10000000000000000000000000000000000
# #
# for i in range(len(y)):
#     if i != 1:
#         left = y[:i]
#         right = y[i:]
#         MSE_a = 0
#         if len(left) > 0:
#             valueLeft = np.sum(left/i)
#             diff_a = valueLeft - left
#             MSE_a = np.sum(pow(diff_a, 2)) / i
#         ##Part 2
#         valueRight = np.sum(right/(len(y)-i))
#         diff_b = valueRight-right
#         MSE_b = np.sum(pow(diff_b,2))/len(right)
#         total_MSE = MSE_a + MSE_b
#         print("Total MSE: ",total_MSE)
#         print("Previous MSE: ", total_MSE_prev)
#         if total_MSE > total_MSE_prev:
#             print("Data is split")
#             splitPoint = i-1
#             print(i-1)
#             break
#         else:
#             total_MSE_prev = total_MSE
#         print("Total MSE pt2:", total_MSE)
#         print("first part: ",left)
#         print("MSE_b :", MSE_b)
#
# #





# valueLeft = np.sum(test[0:])/len(test)
# diff = valueLeft-test
# b = np.sum(pow(diff,2))/len(test)
# print(b)
#
# import tree
# csv_df = pd.read_csv("Income.csv")
# csv_df.sort_values(['Education'], inplace=True)
# y = csv_df['Income']
# z = tree.DecisionTreeRegressor(0,0)
# z.fit(X,y)

#
# total_MSE_prev = 10000000000000000000000000000000000
#         depth = 0
#         splitLeft = None
#         splitRight = None
#         X_split_value = 0
#         total_MSE = 0
#         value = 0
#         splitPoint = 0
#         for i in range(len(y)):
#             if i != 1:
#                 left = y[:i]
#                 right = y[i:]
#                 MSE_a = 0
#                 if len(left) > 0:
#                     valueLeft = np.sum(left / i)
#                     diff_a = valueLeft - left
#                     MSE_a = np.sum(pow(diff_a, 2)) / i
#                 ##Part 2
#                 valueRight = np.sum(right / (len(y) - i))
#
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
#                     xtemp = X.iloc[:, X.column[i]]
#                     X_split_value = (xtemp[i - 2] + xtemp[i - 1]) / 2
#                     print("Split_Value: ", X_split_value)
#                     splitPoint = i - 1
#                     print(i - 1)
#                     splitLeft = X.iloc[:splitPoint]
#                     splitRight = X.iloc[splitPoint:]
#                     depth += 1
#                     break
#                 else:
#                     total_MSE_prev = total_MSE