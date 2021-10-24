import numpy as np
from utility import validate, compute_distances


class KNNBase:

    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, fit_x, fit_y):
        self.X = validate(fit_x)
        self.y = validate(fit_y)

    def predict(self, x):
        x = validate(x)
        predictions = np.zeros((x.shape[0]))
        array_of_distances = compute_distances(self.X, x)
        for i, distances in enumerate(array_of_distances):
            values = np.take(self.y, np.argpartition(distances, self.k)[:self.k])
            predictions[i] = self._getpredictions(values)
        return predictions

    def _getpredictions(self, values):
        raise NotImplementedError()

class KNNClassifier(KNNBase):

    def __init__(self, k=3):
        super().__init__(k)

    def __repr__(self):
        return "KNNClassifier()"

    def _getpredictions(self, values):
        return np.argmax(np.bincount(values))


class KNNRegressor(KNNBase):

    def __init__(self, k=3):
        super().__init__(k)

    def __repr__(self):
        return "KNNClassifier()"

    def _getpredictions(self, values):
        return np.mean(values)
