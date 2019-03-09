import numpy as np
import matplotlib.pyplot as plt
import subprocess
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

def print_scores(rf, _X_train, _X_valid, _y_train, _y_valid):
	print([rmse(rf.predict(_X_train), _y_train), rmse(rf.predict(_X_valid), _y_valid), rf.score(_X_train, _y_train), rf.score(_X_valid, _y_valid), rf.oob_score_ if hasattr(rf, "oob_score_") else ''])
	return rf

def visualize_tree(dt, figsize=(20, 20), feature_names=None):
	export_graphviz(dt, out_file="tree.dot", feature_names=feature_names, rounded=True, filled=True)

	subprocess.call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
	plt.figure(figsize=figsize)
	plt.imshow(plt.imread('tree.png'))


