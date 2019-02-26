import numpy as np 

def mse(y_predicted, y_true):
		# return Mean-Squared Error
	# ((y_hat - y_test) ** 2).mean() -> using numpy
	return np.square(np.subtract(y_predicted, y_true)).mean()

def rmse(y_predicted, y_true):
	# return Root Mean-Squared Error
	return np.sqrt(mse(y_predicted, y_true))


def rsq(y_predicted, y_true):
	# return R^2
	v = np.square(np.subtract(y_predicted, y_true.mean())).mean()
	rsq = 1 - (mse/v)
	return rsq

def print_scores(model, X_train, X_test, y_train, y_test):


def visualize_tree():