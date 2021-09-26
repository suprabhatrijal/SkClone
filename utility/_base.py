import numpy as np


def validate(iterable):
    iterable = np.array(iterable)
    if len(iterable.shape) < 2:
        iterable = iterable.reshape(iterable.shape[0], 1)
    return iterable

def compute_distances(train_data, prediction_data):
    n_samples_train, n_samples_prediction = train_data.shape[0], prediction_data.shape[0]
    distances = np.zeros((n_samples_prediction, n_samples_train))
    for i in range(n_samples_prediction):
        distance = np.zeros(n_samples_train)
        for j in range(n_samples_train):
            distance[j] = np.sum(np.square(train_data[j] - prediction_data[i]))

        distance = np.sqrt(distance)
        distances[i:] = distance
    return distances

