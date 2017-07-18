from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy

seed = 11
numpy.random.seed(seed)

data = numpy.loadtxt("../training_data/2_cost.txt",delimiter="\t")
Xi= data[:,0:8] #input
Yi = data[:,8:9] #output

#standardization
X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))

# create NN
def base_model():
    model = Sequential()
    model.add(Dense(7, input_dim=8,kernel_initializer='normal', activation='relu'))
    # model.add(Dense(8,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
