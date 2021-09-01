import numpy as np

class LinearRegressor:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.slope = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    '''
    Configures the value of slope and intercept according to the line of best fit using the gradient descent 
    algorithm. Works on numpy arrays which are vertically stacked.
    '''

    def fit(self, fit_x, fit_y):
        x = np.array(fit_x)
        y = np.array(fit_y)
        if len(x.shape) < 2:
            x  = x.reshape(x.shape[0], 1)
        n_samples, n_features = x.shape
        y = y.reshape(n_samples,1)

        self.slope = np.zeros(n_features).reshape(n_features, 1)
        self.intercept = 0

        for i in range (self.n_iters):
            y_predicted = (np.dot(x, self.slope) + self.intercept)

            d_slope = -2 * (np.dot(x.T, (y - y_predicted)))
            d_intercept = np.sum(-2 * (y - y_predicted))

            self.slope -= d_slope*self.learning_rate
            self.intercept -= d_intercept*self.learning_rate

    ''' 
    Return the predicted outcome for any dataset according to the configuration of the model as a numpy array which is
    '''
    def predict(self, raw_x):
        x = np.array(raw_x)
        if len(x.shape) < 2:
            x = x.reshape(x.shape[0], 1)
        return np.dot(x, self.slope) + self.intercept








