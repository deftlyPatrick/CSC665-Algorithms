import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
csv_df = pd.read_csv("Melbourne_housing_FULL.csv")
csv_df.head()

def train_test_split(X, y, test_size, shuffle, random_state = None):


    #Ex: 10 subjects -> test_size = 0.2 ; then train = 8 and test = 2
    X_train = round(X * (1 - test_size))
    X_test  = round(X * test_size)
    y_train = round(y * (1 - test_size))
    y_test  = round(y * test_size)

    if shuffle is True:
        np.shuffle(X, y)

    if random_state is None:
        





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

    feat_df = csv_df.drop('Price', axis = 1)
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
    X = feat_df['Date'].values
    feat_df.head()
    rf.fit(feat_df, y)
    return X, y


X, y = preprocess_ver_1(csv_df)

plt.scatter(X, y)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape
