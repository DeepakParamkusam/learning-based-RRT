import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

data = numpy.loadtxt("../training_data/raw/2_control_data.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:10] #output

X_sc = numpy.divide(Xi-Xi.min(axis=0),Xi.max(axis=0)-Xi.min(axis=0))
Y_sc = numpy.divide(Yi-Yi.min(axis=0),Yi.max(axis=0)-Yi.min(axis=0))
X = numpy.divide(X_sc-X_sc.mean(axis=0),X_sc.std(axis=0))
Y = numpy.divide(Y_sc-Y_sc.mean(axis=0),Y_sc.std(axis=0))

#split into training data and validation data
num_data = len(X)
X_train = X[0:int(0.9*num_data),:]
Y_train = Y[0:int(0.9*num_data),:]
X_validate = X[int(0.9*num_data):num_data,:]
Y_validate = Y[int(0.9*num_data):num_data,:]

knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, Y_train)

predictions = knn.predict(X_validate)
actual = Y_validate
print "mse =",mean_squared_error(actual,predictions)

# joblib.dump(knn, '../trained_models/knn_control_2')
