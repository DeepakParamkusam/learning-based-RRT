import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

data = numpy.loadtxt("../training_data/2_cost_10k.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:9] #output

#standardization
X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))
# X = numpy.divide(Xi-Xi.min(axis=0),Xi.max(axis=0)-Xi.min(axis=0))
# Y = numpy.divide(Yi-Yi.min(axis=0),Yi.max(axis=0)-Yi.min(axis=0))

#split into training data and validation data
num_data = len(X)
print num_data
X_train = X[0:int(0.9*num_data),:]
Y_train = Y[0:int(0.9*num_data),:]
X_validate = X[int(0.9*num_data):num_data,:]
Y_validate = Y[int(0.9*num_data):num_data,:]

knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, Y_train)

predictions = knn.predict(X_validate)
actual = Y_validate
# Compute the mean squared error
mse = (((predictions - actual) ** 2).sum()) / len(predictions)
print mse

joblib.dump(knn, '../trained_models/knn_cost_2_10k')
