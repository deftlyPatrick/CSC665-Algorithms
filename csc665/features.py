import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
csv_df = pd.read_csv("Melbourne_housing_FULL.csv")
csv_df.head()

def train_test_split(X, y, test_size, shuffle, random_state = None):


    #Ex: 10 subjects -> test_size = 0.2 ; then train = 8 and test = 2

    if shuffle is True:
        np.random.shuffle(X)
        np.random.shuffle(y)


    trainX = round(len(X) * (1 - test_size))
    testX = round(len(X) * test_size)

    X_train = X[:trainX]
    X_test = X[:testX]

    trainY = round(len(y) * (1 - test_size))
    testY = round(len(y) * test_size)

    y_train = y[:trainY]
    y_test = y[:testY]

    # if random_state is None:

    if random_state is None:
        random.shuffle(X_train)
        random.shuffle(X_test)
        random.shuffle(y_train)
        random.shuffle(y_test)

    return X_train, X_test, y_train, y_test


def create_categories(df, list_columns):
    for i in range(len(list_columns)):
        lst = list_columns[i]
        df[lst] = df[lst].astype('category').cat.codes
    return df

def preprocess_ver_1(csv_df):
    # csv_df = pd.read_csv(csv_file)
    # feat_df = csv_df.drop('Price', axis=1)
    # y = csv_df['Price'].values

    X = csv_df.drop('Price', axis = 1)
    csv_df.shape
    X.shape
    y = csv_df['Price'].values
    y.shape
    rows_labeled_na = csv_df.isnull().any(axis=1)
    rows_with_na = csv_df[rows_labeled_na]
    rows_with_data = csv_df[~rows_labeled_na]
    csv_df.shape, rows_with_na.shape, rows_with_data.shape
    X = rows_with_data.drop('Price', axis=1)
    X.shape
    y = rows_with_data['Price'].values
    y.shape
    suburbs = {}
    for s in X["Suburb"].values:
        if s not in suburbs:
            suburbs[s] = len(suburbs)
    X['Suburb'] = X['Suburb'].replace(suburbs)
    categories_list = ['Type', 'Address', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
    feat_df = create_categories(X, categories_list)
    feat_df.head()
    feat_df['Date'] = pd.to_datetime(feat_df['Date'], infer_datetime_format=True)
    feat_df['Date'] = feat_df['Date'].astype(np.int64)
    X = feat_df['Date'].values
    feat_df.head()
    rf.fit(feat_df, y)
    return X, y


X, y = preprocess_ver_1(csv_df)
print(X)
print(y)



# rf = RandomForestRegressor(n_estimators=1 , random_state=17)
# rf = RandomForestRegressor(n_estimators=5, random_state=17)
# rf = RandomForestRegressor(n_estimators=10 , random_state=17)
# rf = RandomForestRegressor(n_estimators=20, random_state=17)
# rf = RandomForestRegressor(n_estimators=40, random_state=17)
# rf = RandomForestRegressor(n_estimators=80, random_state=17)
# rf = RandomForestRegressor(n_estimators=160, random_state=17)
# rf = RandomForestRegressor(n_estimators=200, random_state=17)



X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25, True)
print(X_train)
print(X_test)
print(y_train)
print(y_test)



# X.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape
