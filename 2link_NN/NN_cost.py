from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy

seed = 11
numpy.random.seed(seed)

data = numpy.loadtxt("cost_C4.txt",delimiter="\t")
Xi= data[:,0:8] #input
Yi = data[:,8:9] #output

#scaling
Xr = Xi.max(axis=0)-Xi.min(axis=0)
Yr = Yi.max(axis=0)-Yi.min(axis=0)
Xs = Xi-Xi.min(axis=0)
Ys = Yi-Yi.min(axis=0)
X = numpy.divide(Xs,Xr)
Y = numpy.divide(Ys,Yr)

# create NN
def base_model():
    model = Sequential()
    model.add(Dense(8, input_dim=8,kernel_initializer='normal', activation='relu'))
    model.add(Dense(8,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
