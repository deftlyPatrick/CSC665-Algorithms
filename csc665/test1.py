import pandas as pd
import numpy as np


def splitFunc(X, y, colm):
    # # X.sort_values(['Education'], inplace=True)
    # df = X
    # df.sort_values(X.columns[colm], inplace=True)
    # # X = csv_df.drop('Income', axis=1)
    # df.reset_index(drop=True, inplace=True)
    # y = df[target]

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
                xtemp = X.iloc[:,colm]
                X_split_value = (xtemp[i-2] + xtemp[i-1])/2
                print("Split_Value: ",X_split_value)
                splitPoint = i-1
                print(i-1)
                splitLeft = X.iloc[:splitPoint]
                splitRight = X.iloc[splitPoint:]

                break
            else:
                total_MSE_prev = total_MSE

            # print("Total MSE pt2:", total_MSE)
            # print("first part: ",left)
            # print("MSE_b :", MSE_b)

    return_val = {'splitPoint':splitPoint, 'MSE': total_MSE, 'Value': value, 'Split Value': X_split_value, 'Left Tree': splitLeft, 'Right Tree': splitRight}
    return return_val
csv_df = pd.read_csv("Income.csv")

X.sort_values(['Education'], inplace=True)

# X.sort_values(X.columns[colm], inplace=True)
# X.reset_index(drop=True, inplace=True)
csv_df.sort_values(csv_df.columns[0], inplace=True)


X = csv_df.drop('Income', axis=1)
y = csv_df['Income']

Z = splitFunc(X, y , 0)


# for i in range(X.shape[1]):
#     X.sort_values(X.columns[i], inplace=True)
#     X.reset_index(drop=True, inplace=True)
#     Z = splitFunc(X, y, i)
#     print(i)