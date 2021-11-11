from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from skclone.metrics import mean_squared_error
from skclone.linear_model import LinearRegressor
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

skclone_model = LinearRegressor()
sklearn_model = LinearRegression()

skclone_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

skclone_predictions = skclone_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test)

print("skclone MSE: ", mean_squared_error(y_test, skclone_predictions))
print("sklearn MSE: ", mean_squared_error(y_test, sklearn_predictions))

fig = plt.figure(figsize=(10,5))
plt.title("Linear Regression Demo")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.scatter(X_test[:, 0], y_test, s=50, cmap="viridis", color="red")


plt.plot(X_test, skclone_predictions, color="green")
plt.legend(['skclone Prediction', "Actual Values"])

plt.savefig("LinearRegression.png")
plt.show()