import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, max_iterations):
        """
        @max_iterations: the maximum number of updating iterations
        to perform before stopping
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = 0
        self.bias = 0
        self.X_bias = 0

    def fit(self, X, y):
        """
        X is an array of input features, dimensions [n_samples, n_features], e.g.
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        y is targets, a single-dim array, [n_samples], e.g.
        [4, 5, 8]
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = np.ones(shape=(n_samples, 1))
        # combines X and bias together
        self.X_bias = np.append(self.bias, X, axis=1)

        # fixes the array problems
        n_features += 1

        self.weights = np.zeros(n_features)

        for i in range(self.max_iterations):
            h = self.predict(X)

            error = h - y

            gradient = np.dot(self.X_bias.T, error)
            gradient /= n_samples
            gradient *= self.learning_rate

            self.weights -= gradient

    def predict(self, X):
        """
         X is an array of input features, dimensions [n_samples, n_features, e.g.
         Returns an Numpy array of real-valued predictions, one for each input, e.g.
         [3.45, 1334.5, 0.94]
        """
        self.bias = np.ones(shape=(X.shape[0], 1))
        self.X_bias = np.append(self.bias, X, axis=1)
        predictions = np.dot(self.X_bias, self.weights)
        return predictions

    def cost_function(self, h, y):
        return 1 / (2 * len(y)) * pow(h-y, 2).sum()


class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = 0
        self.bias = 0
        self.X_bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = np.ones(shape=(n_samples, 1))
        #combines X and bias together
        self.X_bias = np.append(self.bias, X, axis=1)

        #fixes the array problems
        n_features += 1

        self.weights = np.zeros(n_features)

        for i in range(self.max_iterations):
            y_prediction = self.predict_proba(X)

            error = y_prediction - y

            gradient = np.dot(self.X_bias.T, error)
            gradient /= n_samples
            gradient *= self.learning_rate

            self.weights -= gradient


    def predict(self, X):
        """
        X is an array of multiple inputs. Each input is (x1, x2, . . . , xN).
        [[1, 1], [1, 0], [0, 1]
        Returns an Numpy array of classes predicted, e.g. [1, 0, 1, 1].
        Return 1 if probability is >= 0.5, otherwise 0.
        E.g. [1, 1, 0]
        """
        self.bias = np.ones(shape=(X.shape[0], 1))
        self.X_bias = np.append(self.bias, X, axis=1)
        h = self.sigmoid_function(np.dot(self.X_bias, self.weights))
        predictions = [1 if i > 0.5 else 0 for i in h]
        return predictions

    def predict_proba(self, X):
        bias = np.ones(shape=(X.shape[0], 1))
        X_bias = np.append(bias, X, axis=1)
        predictions = np.dot(X_bias, self.weights)
        return self.sigmoid_function(predictions)

    def sigmoid_function(self, X):
        return 1 / (1 + np.exp(-X))