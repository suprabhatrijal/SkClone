from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from skclone.neighbors import KNNClassifier
from skclone.metrics import accuracy_score

iris = load_iris()
X, y, feature= iris.data, iris.target, iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

skclone_model = KNNClassifier(n_neighbors=3)
sklearn_model = KNeighborsClassifier(n_neighbors=3)

skclone_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

skclone_predictions = skclone_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test)

print(f"accuracy score of skclone model: {accuracy_score(y_test, skclone_predictions)*100}%", )
print(f"accuracy score of sklearn model: {accuracy_score(y_test, sklearn_predictions)*100}%", )

fig, axs = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)


axs[0].scatter(X_test[:, 0], X_test[:, 1], c=skclone_predictions, s=50, cmap="viridis")


axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap="viridis")


for ax in axs.flat:
    ax.set(xlabel=feature[0], ylabel=feature[1])
fig.suptitle("KNNClassification Demo")
axs[0].set_title('Skclone Prediction')
axs[0].legend(["bkjbkj","vhhg","gjub"],loc="lower left")
axs[1].set_title('Actual')


plt.savefig("KNN.png")
plt.show()
