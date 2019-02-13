import numpy as np 

def mse(y_predicted, y_true)
	#return Mean-Squared Error
	#((y_hat - y_test) ** 2).mean() -> using numpy 
	return np.square(np.subtract(y_predicted - y_true)).mean()

def rmse(y_predicted, y_true)
	#return Root Mean-Squared Error
	return np.sqrt(np.square(np.subtract(y_predicted - y_true)).mean())


def rsq(y_predicted, y_true)
	#return R^2 
	return np.square(y_predicted), np.square(y_true)