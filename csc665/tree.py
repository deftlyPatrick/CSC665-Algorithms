import pandas as pd
import numpy as np

class DecisionTreeRegressor():
	def __init__(self, max_depth, min_samples_leaf):
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf

	def __repr__(self):
		raise NotImplementedError()

	def fit(self, X: pd.DataFrame, y: np.array):
		self.indices = range(N)
		self.N = X.shape[0]

		self.internal_fit(X, y, range(X.shape[0]), 0)

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
		