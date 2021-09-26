import numpy as np
from utility import validate, compute_distances


class KNNClassifier:

    def __init__(self, k = 3):
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
        for i in range(len(array_of_distances)):
            distances = array_of_distances[i]
            values = np.take( self.y,np.argpartition(distances, self.k)[:self.k])
            counts = np.bincount(values)
            predictions[i] =np.argmax(counts)

        return predictions

