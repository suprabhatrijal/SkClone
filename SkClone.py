import numpy as np

class LinearRegressor:
    def __repr__(self):
        return "Learning rate: {0}".format(self.learning_rate)

    def __init__(self, slope=1, intercept=2, learning_rate=0.0001):
        self.slope = slope
        self.intercept = intercept
        self.learning_rate = learning_rate

    #gives the derivative of the loss function
    def rss_derivative(self, x, y):
        slope = self.slope
        intercept = self.intercept
        d_slope = 0
        d_intercept = 0
        d_slope_array = 2 * -x * (y - (slope * x + intercept))
        d_intercept_array = -2 * (y - (slope * x + intercept))
        for i in range(0, len(d_slope_array)):
            d_slope += d_slope_array[i][0]
            d_intercept += d_intercept_array[i][0]
        return[d_slope, d_intercept]



    #fits the model according to the data
    def fit(self, fit_x, fit_y):
        fit_x, fit_y = np.vstack(fit_x), np.vstack(fit_y)
        derivatives = self.rss_derivative(fit_x, fit_y)
        while abs(derivatives[0]) >= 0.001 or abs(derivatives[1]) >= 0.001:
            step_slope = derivatives[0] * self.learning_rate
            step_intercept = derivatives[1] * self.learning_rate
            self.slope -= step_slope
            self.intercept -= step_intercept
            derivatives = self.rss_derivative(fit_x, fit_y)

    #gives the prediction
    def predict(self, x):
        pred_x = np.vstack(x)
        pred_y = self.slope*pred_x + self.intercept
        return pred_y


#Class with functions to check the accuracy of the model
class Metric:
    def rss(self, y, prediction):
        prediction = np.vstack(prediction)
        y = np.vstack(y)
        rss = np.sum((y - prediction) ** 2)



        return rss




