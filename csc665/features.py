


def train_test_split(X, y, test_size, shuffle, random_state = NONE):
	#X, y - features and the target variable
	#test size - between 0 and 1 - how much to allocate to the test set; the rest goes to the train set. 
	#shuffle - if True, shuffle the dataset, otherwise not. 
	#random_state, integer; if None, the results are random, otherwise fixed to a given seed. 
	#Example


create_catories(df, list_columns)
	#converts values in place, in-place, in the columns passed in the list_columns to numerical values. 
	#Follow the same approach: "string" -> category ->code.
	#Replace values in df, in-place

X,y = preprocess_ver_1_(csv_df)
	#Apply the feature transformation steps to the dataframe, return new x and y for the entire dataset. Do not modify the original csv_df.
	#Remove all rows with NA values
	#Convert datatime to a number
	#Convert all strings to numbers
	#Split the dataframe into X and y and return these. 

