from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from skclone.cluster import KMeansCluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


iris = load_iris()
X, y , feature= iris.data, iris.target, iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)


model = KMeansCluster(n_clusters=3)
model1 = KMeans(n_clusters=3)

model.fit(X_train)
model1.fit(X_train)

predictions = model.predict(X_test)
predictions1 = model1.predict(X_test)


fig, axs = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True, sharey=True)


axs[0].scatter(X_test[:, 0], X_test[:, 1], c=predictions, s=50, cmap="viridis")
axs[0].set_title('Skclone Prediction')


axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap="viridis")
axs[1].set_title('Actual')

for ax in axs.flat:
    ax.set(xlabel=feature[0], ylabel=feature[1])


fig.suptitle("KMeansClassification Demo")
plt.savefig("KMeans.png")
plt.show()

