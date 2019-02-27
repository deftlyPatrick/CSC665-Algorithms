import pandas as pd
import numpy as np
import metrics
import features
#X = row
#y = column
# csv_df = pd.read_csv("education.csv")
# csv_df.head()


class DecisionTreeRegressor():

	def __init__(self, max_depth, min_samples_leaf):
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf

	def __repr__(self):
		csv_df = pd.read_csv("education.csv")
		csv_df.sort_values('Education')
		print(csv_df.head())
		X = csv_df
		y = csv_df['Income'].values

		DecisionTreeRegressor.fit(self, X, y)
		raise NotImplementedError()

	def fit(self, X: pd.DataFrame, y: np.array):
		self.N = X.shape[0]
		self.indices = range(self.N)

		self.internal_fit(X, y, self.indices, 0)

	def internal_fit(self, X, y, indices, depth):
		self.X = X
		self.y = y
		self.depth = depth

		self.value = None
		self.mse = None
		self.N = None

		self.split_col = None
		self.split_val = None
		self.split_mse = None

		self.left = None
		self.right = None

		self.split()

	def split(self):
		for i in range(self.X.shape[1]):
			find_best_split(i)

	def find_best_split(self, i):
		self.split_col = i
		X = self.X.values[self.indices, i]
		y = self.y[indices]

		for i in range(self.N):
			left = X <= X[i]
			right = X > X[i]

		self.left = DecisionTreeRegressor()
		self.left.internal_fit()
		self.right = DecisionTreeRegressor()
		self.right.internal_fit()

	def predict(self, X: pd.DataFrame):
		pass


	def score(self, X: pd.DataFrame, y: np.array):
		pass
