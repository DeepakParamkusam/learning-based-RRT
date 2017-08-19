import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

data = numpy.loadtxt("../training_data/2_control_data_fulltraj_10k.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:48] #output

#standardization
X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))
# X = numpy.divide(Xi-Xi.min(axis=0),Xi.max(axis=0)-Xi.min(axis=0))
# Y = numpy.divide(Yi-Yi.min(axis=0),Yi.max(axis=0)-Yi.min(axis=0))
a = Xi.mean(axis=0)
# print a
b = Xi.std(axis=0)
c = Yi.mean(axis=0)
d = Yi.std(axis=0)

#split into training data and validation data
num_data = len(X)
X_train = X[0:int(0.9*num_data),:]
print len(X_train)
Y_train = Y[0:int(0.9*num_data),:]
X_validate = X[int(0.9*num_data):num_data,:]
Y_validate = Y[int(0.9*num_data):num_data,:]

knn = KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predictions = knn.predict(X_validate)
actual = Y_validate
# Compute the mean squared error
mse = (((predictions - actual) ** 2).sum()) / len(predictions)
print mse

iState = numpy.zeros(8)
iState[0] = 0.2411
iState[1] =	5.1476
iState[2] = 1.702
iState[3] = -11.6195
iState[4] = 5.1472
iState[5] = 1.2570
iState[6] = -18.9352
iState[7] = 2.8552

inp = numpy.divide(iState-a,b)
out = knn.predict(inp)
out= out*d +c
print out

# joblib.dump(knn, '../trained_models/knn_control_ft_2_10k_test')
