from keras.models import Sequential
from keras.layers import Dense
import numpy

data = numpy.loadtxt("../training_data/2_control_data.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:10] #output

#standardization
X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))

#split into training data and validation data
num_data = int(len(X)/8.0)
X_train = X[0:int(0.75*num_data),:]
Y_train = Y[0:int(0.75*num_data),:]
X_validate = X[int(0.75*num_data):num_data,:]
Y_validate = Y[int(0.75*num_data):num_data,:]

# create NN
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=10)

scores = model.evaluate(X_validate, Y_validate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('../trained_models/NN_control_2_H10H4')
