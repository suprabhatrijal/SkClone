import numpy as np

def mean_squared_error( y, prediction):
    if len(y.shape) < 2:
        y = y.reshape(y.shape[0], 1)
    if len(prediction.shape) < 2:
        prediction = prediction.reshape(prediction.shape[0], 1)
    return (1/y.shape[0])*np.sum((y-prediction)**2)
