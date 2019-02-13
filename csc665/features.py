from sklearn.utils import shuffle
import numpy as np 
import pandas as pd


def train_test_split(X, y, test_size, shuffle, random_state = NONE):
	#Requirements

	#X, y - features and the target variable
	#test size - between 0 and 1 - how much to allocate to the test set; the rest goes to the train set. 
	#shuffle - if True, shuffle the dataset, otherwise not. 
	#random_state, integer; if None, the results are random, otherwise fixed to a given seed. 
	#Example
		#X_train & X_test = feature for the training and testing data
		#y_train & y_train = labels for the training and testing data
		#X_train, X_test, y_train, y_test = train_test_split(feat_df, y, 0.3, True, 12)
	test

	if shuffle = True:
		shuffle(X, y)
	else:
		print("Shuffle is false.")

	if random_state = NONE:	
		seed = np.random(X,y)
	else:



create_catories(df, list_columns)
	#Requirements

	#converts values in place, in-place, in the columns passed in the list_columns to numerical values. 
	#Follow the same approach: "string" -> category ->code.
	#Replace values in df, in-place

	#iterate through every row and column and change column -> int 
	for list_columns in df.itertuples():
		inPlace = df.pd.to_numeric(s, errors = 'coerce')






X, y = preprocess_ver_1_(csv_df)
	#Requirements

	#Apply the feature transformation steps to the dataframe, return new x and y for the entire dataset. Do not modify the original csv_df.
	#Remove all rows with NA values
	#Convert datetime to a number
	#Convert all strings to numbers
	#Split the dataframe into X and y and return these. 

	#Apply the feature transformation steps to the dataframe, 
	#return new x and y for the entire dataset. Do not modify the original csv_df.
	df = pd.DataFrame(data = {'X': X, 'y': y}
	df.transform(X, y)

	#Remove all rows with NA values
	index = np.arange(X, y)
	csv_df = pd.DataFrame(index = index)

	#Convert datetime to a number
	csv_df['Date'] = pd.to_datetime(csv_df['Date'], infer_datetime_format = True)
	csv_df['Date'] = csv_df['Date'].astype(np.int64)

	#Convert all strings to numbers


	#Split the dataframe into X and y and return these.



	
