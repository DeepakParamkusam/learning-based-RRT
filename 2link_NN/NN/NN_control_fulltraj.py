from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.externals import joblib
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

data = "../training_data/control_ft_2_10k_scaled"

#load data
X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)
print X_train.shape

# create NN
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(40,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=904)

scores = model.evaluate(X_validate, Y_validate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('../trained_models/NN_control_H8')
