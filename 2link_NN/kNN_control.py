import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

data = numpy.loadtxt("control_data_C2.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:10] #output

#scaling
Xr = Xi.max(axis=0)-Xi.min(axis=0)
Yr = Yi.max(axis=0)-Yi.min(axis=0)
Xs = Xi-Xi.min(axis=0)
Ys = Yi-Yi.min(axis=0)
X = numpy.divide(Xs,Xr)
Y = numpy.divide(Ys,Yr)

#split into training data and validation data
num_data = int(len(X)/8.0)
X_train = X[0:int(0.75*num_data),:]
Y_train = Y[0:int(0.75*num_data),:]
X_validate = X[int(0.75*num_data):num_data,:]
Y_validate = Y[int(0.75*num_data):num_data,:]

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, Y_train)

predictions = knn.predict(X_validate)
actual = Y_validate
# Compute the mean squared error
mse = (((predictions - actual) ** 2).sum()) / len(predictions)
print mse

joblib.dump(knn, 'knn_control_C2')
