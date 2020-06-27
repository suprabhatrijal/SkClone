from SkClone import LinearRegressor
import pandas as pd


df = pd.read_csv('demo/linear regression/demo_dataset.csv')

test_model = LinearRegressor()

test_model.fit(df["x"][:20], df["y"][:20])

prediction  = test_model.predict(df["x"][20:])
print(prediction)






