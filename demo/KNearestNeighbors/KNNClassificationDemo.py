from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from neighbors import KNNClassifier
from metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

skclone_model = KNNClassifier(k=3)
sklearn_model = KNeighborsClassifier(n_neighbors=3)

skclone_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

skclone_predictions = skclone_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test)

print(f"accuracy score of skclone model: {accuracy_score(y_test, skclone_predictions)*100}%", )
print(f"accuracy score of sklearn model: {accuracy_score(y_test, sklearn_predictions)*100}%", )
