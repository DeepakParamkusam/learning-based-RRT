from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy

seed = 11
numpy.random.seed(seed)

data = numpy.loadtxt("../training_data/2_cost_10k.txt",delimiter="\t")
Xi= data[:,0:8] #input
Yi = data[:,8:10] #output

#standardization
Xs = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
Ys = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))

#scaling
X = numpy.divide(Xs-Xs.min(axis=0),Xs.max(axis=0)-Xs.min(axis=0))
Y = numpy.divide(Ys-Ys.min(axis=0),Ys.max(axis=0)-Ys.min(axis=0))

#split into training data and validation data
num_data = len(X)
# print len(X)
X_train = X[0:int(0.9*num_data),:]
Y_train = Y[0:int(0.9*num_data),:]
X_validate = X[int(0.9*num_data):num_data,:]
Y_validate = Y[int(0.9*num_data):num_data,:]

# create NN
model = Sequential()
model.add(Dense(4, input_dim=8, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(6, activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=10)

scores = model.evaluate(X_validate, Y_validate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
