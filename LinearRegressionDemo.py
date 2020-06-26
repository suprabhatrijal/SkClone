from SkClone import LinearRegressor


train = [(8,3),(18,10),(11,4),(13,6),(14,8),(22,12),(12,5),(9,4),(20,9),(25,14),(17,2)]
prediction_x = [5,10,20,11,32]
x =  [points[0] for points in train]
y = [points[1] for points in train]


test_model = LinearRegressor()
test_model.fit(x,y)


