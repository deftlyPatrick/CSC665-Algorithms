import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz

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
	return mse(X_train, y_train), mse(X_test, y_test), rsq(X_train, y_train), rsq(X_test, y_test)

def visualize_tree(dt, figsize(20, 20), feature_names=None):
	export_graphviz(dt, out_file="tree.dot", feature_names=feature_names, rounded=True, filled=True)
	subprocess.call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
	plt.figure(figsize = figsize)
	plt.imshow(plt.imread('tree.png'))
