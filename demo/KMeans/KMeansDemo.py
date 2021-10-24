# import all the libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from utility import validate
from cluster import KMeansCluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# load the data
iris = load_iris()
# splitting the data into features and target
X, y = iris.data, iris.target
# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)

# formatting the data into the required format
X_train = validate(X_train)

model = KMeansCluster(n_clusters=3)
model1 = KMeans(n_clusters=3)

model.fit(X_train)
model1.fit(X_train)

predictions = model.predict(X_test)
predictions1 = model1.predict(X_test)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(X_test[:, 0], X_test[:, 1], c=predictions, s=50, cmap="viridis")
ax1.set_title('Skclone Prediction')


ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap="viridis")
ax2.set_title('Actual')

plt.show()

