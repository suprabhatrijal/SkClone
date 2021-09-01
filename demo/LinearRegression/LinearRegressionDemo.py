from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.make_regression(n_samples=200, n_features=10, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)



from metrics import mean_squared_error
from linear_model import LinearRegressor

skclone_model = LinearRegressor(learning_rate=0.001)
sklearn_model = LinearRegression()

skclone_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

skclone_predictions = skclone_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test)


print("skclone MSE: ", mean_squared_error(y_test, skclone_predictions))
print("sklearn MSE: ", mean_squared_error(y_test, sklearn_predictions))





