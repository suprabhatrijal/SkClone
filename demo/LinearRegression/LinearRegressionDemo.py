from SkClone import LinearRegressor
import pandas as pd


df = pd.read_csv('demo_dataset.csv')

test_model = LinearRegressor()
test_model.fit(df["x"], df["y"])
prediction  = test_model.predict(df["x"])

print(prediction)






