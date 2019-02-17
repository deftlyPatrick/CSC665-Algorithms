import numpy as np
import pandas as pd
from sklearn import utils
import random
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# csv_df = pd.read_csv("Melbourne_housing_FULL.csv")
# csv_df.head()



def train_test_split(X, y, test_size, shuffle, random_state = None):


    #Ex: 10 subjects -> test_size = 0.2 ; then train = 8 and test = 2
    #Random state only used in shuffle

    if shuffle is True:
        indices = np.arange(y.shape[0])
        random_index = utils.shuffle(indices, n_samples=len(indices), random_state=random_state)
        # shuffle_x = X.iloc[[random_index],:]
        # shuffle_x = X.reindex(random_index)
        shuffle_y = y[random_index]
        X.reset_index(drop=True, inplace=True)
        shuffle_x = X.reindex(random_index)
        # shuffle_x = X.reindex(random_index).dropna()

        # print(shuffle_x)
        # print(shuffle_y)

    # Multiply the test_size to the length of x to determine how many number to split between test and train
    x_size = round(len(shuffle_x) * test_size)

    X_train = shuffle_x[:-x_size]
    X_test = shuffle_x[-x_size:]

    y_train = shuffle_y[:-x_size]
    y_test = shuffle_y[-x_size:]

    return X_train, X_test, y_train, y_test


# def shuffle_data(X, y):
#     x_length = len(X)
#     for i in range(x_length-1):
#         swap_data(X, y, i, random.randrange(i, x_length))
#         return X, y
#
#
# def swap_data(X, y, index, random_index):
#     # Swap the index and replace with random index
#     X[index], X[random_index] = X[random_index], X[index]
#     y[index], y[random_index] = y[random_index], y[index]
#
#

def create_categories(df, list_columns):
    for i in range(len(list_columns)):
        lst = list_columns[i]
        df[lst] = df[lst].astype('category').cat.codes
    return df


def preprocess_ver_1(csv_df):
    # csv_df = pd.read_csv(csv_file)

    # feat_df = csv_df.drop('Price', axis=1)
    # y = csv_df['Price'].values

    feat_df = csv_df.drop('Price', axis=1)
    csv_df.shape
    feat_df.shape
    y = csv_df['Price'].values
    y.shape
    rows_labeled_na = csv_df.isnull().any(axis=1)
    rows_with_na = csv_df[rows_labeled_na]
    rows_with_data = csv_df[~rows_labeled_na]
    csv_df.shape, rows_with_na.shape, rows_with_data.shape
    feat_df = rows_with_data.drop('Price', axis=1)
    feat_df.shape
    y = rows_with_data['Price'].values
    y.shape
    suburbs = {}
    for s in feat_df["Suburb"].values:
        if s not in suburbs:
            suburbs[s] = len(suburbs)
    feat_df['Suburb'] = feat_df['Suburb'].replace(suburbs)
    categories_list = ['Type', 'Address', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
    feat_df = create_categories(feat_df, categories_list)
    feat_df.head()
    feat_df['Date'] = pd.to_datetime(feat_df['Date'], infer_datetime_format=True)
    feat_df['Date'] = feat_df['Date'].astype(np.int64)
    # feat_df = feat_df['Date'].values
    # rf.fit(feat_df, y)
    return feat_df, y


# X, y = preprocess_ver_1(csv_df)
#
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=5)
# print(X.shape)
# print(X_train1.shape)
# print(X_test1.shape)
# print(y_train1.shape)
# print(y_test1.shape)
# print("\n")
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, 0.25, True, 5)
# print(X_train2)
# print(X_test2)
# print(y_train2)
# print(y_test2)
# print("\n")
# X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, 0.25, True, None)
# print(X_train3)
# print(X_test3)
# print(y_train3)
# print(y_test3)
# print("\n")
# X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, 0.25, True, None)
# print(X_train4)
# print(X_test4)
# print(y_train4)
# print(y_test4)
# print("\n")

# X_train, X_test, y_train, y_test = train_test_split(x_1, y_1, 0.25, True, None)



# X.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape

