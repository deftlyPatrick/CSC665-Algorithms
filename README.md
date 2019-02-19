# CSC 665 - Project 1 

## Part 1: Creating a library 

### features.py

`def train_test_split(X, y, test_size, shuffle, random_state = None)`

Parameters:
- X, y - features and the target variable
- test_size - between 0 and 1 - how much to allocate to the test size; the rest goes to the train set. 
- shuffle - if True, shuffle the dataset, otherwise not.
- random_state, integer; if None, then results are random, otherwise fixed to a given seed.
Example:
	`X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3, True, 12)`

- - - -

`def create_categories(df, list_columns)`

- Create values, in-place, in the columns passed in the list_columns to numerical values. Follow the same approach: "string" -> category -> code
- Replace values in df, in-place

- - - -

`def preprocess_ver_1(csv_df)`

Apply the features transformation steps to the dataframe, return new X and y for the entire dataset. Do not modify the original csv_df.
	- Remove all rows with NA values
	- Convert datatime to a number
	- Convert all strings to numbers.
	- Split the dataframe into X and y and return these.

- - - -

### metrics.py

`def mse(y_predicted, y_true)`
	- return Mean-Squared Error

`def rmse(y_predicted, y_true)`
	- return Root Mean-Squared Error

`def rsq(y_predicted, y_true)`
	- return R^2

## Part 2: Using library to Jupyter Notebook

To terminal, type in
	`source activate py36
	 jupyter notebook`

Train vs Test dataset 
Plot for trees from 1 to 200, in interval of 5 

Code for determining train score
`rf.score(X_train, y_train)`

Code for determining test score
`test_score_mean = mse(y_predicted, y_true)
test_score_sq = rsq(y_predicted, y_true)
score = 1 - mse/rsq`

