from SkClone import LinearRegressor
import pandas as pd
import time

df = pd.read_csv('demo_dataset.csv')

test_model = LinearRegressor()
start = time.time()
test_model.fit(df["x"], df["y"])
stop =time.time()
prediction  = test_model.predict(df["x"][20:])


print(prediction)
print(stop-start)





