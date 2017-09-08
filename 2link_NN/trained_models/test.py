from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.externals import joblib
from keras.models import load_model
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

data = "../training_data/control_ft_2_10k_scaled"

#load data
X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)
nn = load_model('NN_control_H5')
knn = joblib.load('knn_control_ft_2_10k_scaled')

test_case = X_train[0]
out_knn = knn.predict(test_case)
out_nn = nn.predict(test_case.reshape([1,8]))

print out_knn
print out_nn
print Y_train[0]

print out_knn*coeff[3]+coeff[2]
print out_nn*coeff[3]+coeff[2]
print Y_train[0]*coeff[3]+coeff[2]
print coeff[3]
print coeff[2]
