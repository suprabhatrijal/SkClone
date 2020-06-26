class LinearRegressor:

    def __repr__(self):
        return "Learning rate: {0}".format(self.learning_rate)

    def __init__(self):
        self.slope = 1
        self.intercept = 2
        self.learning_rate = 0.0001

    #gives the derivative of the loss function
    def rss_derivative(self, x, y):
        d_slope = 0
        d_intercept = 0
        length = len(x)
        slope = self.slope
        intercept = self.intercept
        for i in range(0, length):
            x_point = x[i]
            y_point = y[i]
            d_slope += 2 * -x_point * (y_point - (slope * x_point + intercept))
            d_intercept += -2 * (y_point - (slope * x_point + intercept))

        return [d_slope, d_intercept]
    #fits the model according to the data
    def fit(self, fit_x, fit_y):
        derivatives = self.rss_derivative(fit_x, fit_y)
        while abs(derivatives[0]) >= 0.001 or abs(derivatives[1]) >= 0.001:
            step_slope = derivatives[0] * self.learning_rate
            step_intercept = derivatives[1] * self.learning_rate
            self.slope -= step_slope
            self.intercept -= step_intercept
            derivatives = self.rss_derivative(fit_x, fit_y)

    #gives the prediction
    def predict(self, pred_x):
        length = len(pred_x)
        pred_y = []
        for i in range(0, length):
            pred_y.append(self.slope*pred_x[i] + self.intercept)

        return pred_y


#Class with functions to check the accuracy of the model
class Metric:
    def rss(self, points, prediction):
        rss = 0
        for i in range(0, len(prediction)):
            y = points[i][1]
            y_predict = prediction[i]

            rss += (y - y_predict) ** 2
        return rss




