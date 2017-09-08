import numpy
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

data = "../training_data/cost_2_10k_std"
num_neigh = 9

#load data
X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)

#train kNN
knn = KNeighborsRegressor(n_neighbors=num_neigh)
knn.fit(X_train, Y_train)

#validate and compute the mean squared error
predictions = knn.predict(X_validate)
actual = Y_validate
print "mse =",mean_squared_error(Y_validate,predictions)

#save kNN
joblib.dump(knn, '../trained_models/knn_cost_2_10k_std')
