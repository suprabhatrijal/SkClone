from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


from linear_model import LogisticRegressor
from metrics import accuracy_score

Dataset = datasets.load_breast_cancer()
X, y = Dataset.data, Dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

skclone_model = LogisticRegressor()
sklearn_model = LogisticRegression(max_iter=10000)

skclone_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

skclone_predictions = skclone_model.predict(X_test)
sklearn_predictions = sklearn_model.predict(X_test)

print(f"accuracy score of skclone model: {accuracy_score(y_test, skclone_predictions)*100}%", )
print(f"accuracy score of sklearn model: {accuracy_score(y_test, sklearn_predictions)*100}%", )
