import numpy as np

class LinearRegressor:
    def __repr__(self):
        return "Learning rate: {0}".format(self.learning_rate)

    def __init__(self, slope=1, intercept=2, learning_rate=0.0001):
        self.slope = slope
        self.intercept = intercept
        self.learning_rate = learning_rate

    ''' 
    Returns the derivative of the loss function w.r.t slope and intercept in a list where the first item is the 
    derivative w.r.t. slope and the second item is the derivative w.r.t. intercept
    '''
    def rss_derivative(self, x, y):
        slope = self.slope
        intercept = self.intercept
        # d_slope is the the value of derivative of the loss function w.r.t. slope
        d_slope= np.sum(2 * -x * (y - (slope * x + intercept)))
        # d_intercept is the the value of derivative of the loss function w.r.t. intercept
        d_intercept = np.sum(-2 * (y - (slope * x + intercept)))
        return [d_slope, d_intercept]

    '''
    Configures the value of slope and intercept according to the line of best fit using the gradient descent 
    algorithm. Works on numpy arrays which are vertically stacked.
    '''
    def fit(self, fit_x, fit_y):
        fit_x, fit_y = np.vstack(fit_x), np.vstack(fit_y)
        derivatives = self.rss_derivative(fit_x, fit_y)
        while abs(derivatives[0]) >= 0.001 or abs(derivatives[1]) >= 0.001:
            step_slope = derivatives[0] * self.learning_rate
            step_intercept = derivatives[1] * self.learning_rate
            self.slope -= step_slope
            self.intercept -= step_intercept
            derivatives = self.rss_derivative(fit_x, fit_y)

    ''' 
    Return the predicted outcome for any dataset according to the configuration of the model as a numpy array which is
    vertically stacked
    '''
    def predict(self, x):
        pred_x = np.vstack(x)
        pred_y = self.slope*pred_x + self.intercept
        return pred_y


# Class with functions to check the accuracy of the model
class Metric:
    def rss(self, y, prediction):
        prediction = np.vstack(prediction)
        y = np.vstack(y)
        rss = np.sum((y - prediction) ** 2)
        return rss




