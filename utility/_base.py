import numpy as np

# function to validate the data
def validate(iterable):
    iterable = np.array(iterable)
    if len(iterable.shape) < 2:
        iterable = iterable.reshape(iterable.shape[0], 1)
    return iterable


def compute_distances(train_data, prediction_data):
    n_samples_train, n_samples_prediction = train_data.shape[0], prediction_data.shape[0]
    distances = np.zeros((n_samples_prediction, n_samples_train))
    for i, prediction_value in enumerate(prediction_data):
        distance = np.zeros(n_samples_train)
        for j, train_value in enumerate(train_data):
            distance[j] = np.sum(np.square(train_value - prediction_value))
        distance = np.sqrt(distance)
        distances[i:] = distance
    return distances
