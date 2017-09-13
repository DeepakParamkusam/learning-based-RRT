import sys
import numpy
from numpy.random import seed
from keras.layers import Dense
from keras.models import Sequential
from sklearn.externals import joblib
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

if len(sys.argv) == 3:
    num_data = str(sys.argv[1])
    HL1 = int(sys.argv[2])
elif len(sys.argv) == 4:
    num_data = str(sys.argv[1])
    HL1 = int(sys.argv[2])
    HL2 = int(sys.argv[3])
else:
    print 'Incorrect no. of arguments'
    exit()

data = '../training_data/cost_2_' + num_data + 'k_scaled'

#load data
X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)

# create NN
model = Sequential()
model.add(Dense(HL1, input_dim=8, activation='relu'))
if len(sys.argv) == 4:
    model.add(Dense(HL2,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=902)

scores = model.evaluate(X_validate, Y_validate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if len(sys.argv) == 4:
    model.save('../trained_models/nn/cost/NN_cost_' + num_data + 'k_' + str(HL1) + '_' + str(HL2))
else:
    model.save('../trained_models/nn/cost/NN_cost_' + num_data + 'k_' + str(HL1))
