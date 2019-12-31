# CSC 665 - Basic Machine Learning Algorithms

- - - -

## Algorithms Included:

### Decision Tree Classifer 
### Decision Tree Regressor
### Random Forest Regressor 
### Linear Regression 
### Logistic Regression
### Perceptron (AND, OR, XNOR, XOR)
### Single Layer Neural Networks
### Multi Layer Neural Networks


- - - -
## How to use library to Jupyter Notebook

To terminal, type in
- `source activate py36`
- `jupyter notebook`

Latest version (as of Dec 2019):
- `export PATH="/Users/user_name/opt/anaconda3/bin:$PATH"`
- `source activate py37`
- `jupyter notebook`

- - - - 

Train vs Test dataset 
- Plot for trees from 1 to 200, in interval of 5 

Code for determining train score
- `rf.score(X_train, y_train)`

Code for determining test score
- `test_score_mean = mse(y_predicted, y_true)`
- `score = 1 - mse/rsq`

- - - - 

## Source Code 

### features.py

`def train_test_split(X, y, test_size, shuffle, random_state = None)`

Parameters:
- X, y - features and the target variable
- test_size - between 0 and 1 - how much to allocate to the test size; the rest goes to the train set. 
- shuffle - if True, shuffle the dataset, otherwise not.
- random_state, integer; if None, then results are random, otherwise fixed to a given seed.
- Example:
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

`def accuracy_score(y_pred, y_actual)`
-returns the accuracy between the test and training data

`def rsq(y_predicted, y_true)`
- return R^2

`def print_scores(rf, _X_train, _X_valid, _y_train, _y_valid)`
- returns 

- - - - 

### tree.py - Decision Tree Regressor/Classifier

###Decision Tree Regressor

`def split(indices: np.array)`
- stores the predicted value that was determined using the smallest MSE (stores the smallest MSE)

`def find_best_split(col_index, indices)`
- compares every MSE and stores the best one possible

`def calc_mse`
- return MSE for that certain index

###Decision Tree Classifer

`def split(indices: np.array)`
- predicts the object that is more likely to happen

`def find_best_split(col_index, indices)`
- compares every Gini weight and stores the best one possible

`def calc_gini`
- return gini for that certain index

- - - - 

### perception.py - Perceptions - basic neural networks 

`class PerceptronLayer`
- uses the step-wise activation function to determine if 1 or 0

`def forward(x)` 
- uses gradient descent algorithm then uses the step-wise function


`def step_function(x)`
- takes in the gradent descent results and determine if its 1 or 0


`class BooleanFactory`
- create weights for every type of perceptron (AND, OR, NOT, XNOR, XOR)

Weights for perceptrons

Single-layer (linear) = AND, OR, NOT
AND = `np.array([-30, 20, 20])`
OR = `np.array([-10, 20, 20])`
NOT = `np.array([1, -1])`

Multi-layer (non-linear) = XNOR, XOR
XNOR = 
```		
		#and
		w1 = np.array([-30, 20, 20])

		#not x1 and not x2
		w2 = np.array([10, -20, -20])

		#or
		w3 = np.array([-10, 20, 20])
```
XOR = 
```
		#or
		w1 = np.array([-10, 20, 20])

		#nand
		w2 = np.array([1.5, -1, -1])

		#and
		w3 = np.array([-30, 20, 20])
```

### graph.py - Neural Networks (Forward/Backward propagation)

Still in progress - being improved

