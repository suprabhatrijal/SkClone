import numpy as np


class BaseRegressor:
    def __init__(self, learning_rate=0.0001, n_iters=1000):
        self.slope = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def _approximation(self, x, slope, intercept):
        raise NotImplementedError()

    def _predict(self, x, slope, intercept):
        return NotImplementedError()

    def _validate(self, iterable):
        iterable = np.array(iterable)
        if len(iterable.shape) < 2:
            iterable = iterable.reshape(iterable.shape[0], 1)
        return iterable

    def predict(self, x):
        x = self._validate(x)
        return self._predict(x, self.slope, self.intercept)

    def fit(self, fit_x, fit_y):
        x = self._validate(fit_x)
        y = self._validate(fit_y)

        n_samples, n_features = x.shape

        self.slope = np.zeros((n_features, 1))
        self.intercept = 0

        for i in range(self.n_iters):

            y_predicted = self._approximation(x, self.slope, self.intercept)

            d_slope = (1/n_samples) * -2 * (np.dot(x.T, (y - y_predicted)))
            d_intercept = (1/n_samples) * np.sum(-2 * (y - y_predicted))

            self.slope -= d_slope*self.learning_rate
            self.intercept -= d_intercept*self.learning_rate


class LinearRegressor(BaseRegressor):

    def __init__(self, learning_rate=0.01, n_iters=1000):
        super().__init__(learning_rate, n_iters)
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def __repr__(self):
        return "LinearRegressor()"

    def _approximation(self, x, slope, intercept):
        return np.dot(x, slope) + intercept

    def _predict(self, x, slope, intercept):
        return self._approximation(x, slope, intercept)


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


class LogisticRegressor(BaseRegressor):

    def __init__(self, learning_rate=0.0001, n_iters=10000):
        super().__init__(learning_rate, n_iters)
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def __repr__(self):
        return "LogisticRegressor()"

    def _approximation(self, x, slope, intercept):
        linear_model = np.dot(x, slope) + intercept
        return _sigmoid(linear_model)

    def _predict(self, x, slope, intercept):
        y_predicted = self._approximation(x, slope, intercept)
        y_predicted_class = np.fromiter((1 if i > 0.5 else 0 for i in y_predicted), dtype=int)
        return y_predicted_class
