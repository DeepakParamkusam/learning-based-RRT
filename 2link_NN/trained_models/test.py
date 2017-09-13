import sys
import numpy
from numpy.random import seed
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.externals import joblib
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

def test_control_knn(num_data, type_data, num_neigh):
    data = '../training_data/control_ft_2_'+ str(num_data) + 'k_' + type_data
    #load data
    X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)
    knn = joblib.load('knn/control/knn_control_ft_2_' + str(num_data) + 'k_' + type_data + '_' + str(num_neigh))
    return X_train,Y_train,coeff,knn

def test_cost_knn(num_data, type_data, num_neigh):
    data_cost = '../training_data/cost_2_'+ str(num_data) + 'k_' + type_data
    #load data
    X_train_cost,Y_train_cost,X_validate_cost,Y_validate_cost,coeff_cost = joblib.load(data_cost)
    knn_cost = joblib.load('knn/cost/knn_cost_2_' + str(num_data) + 'k_' + type_data + '_' + str(num_neigh))
    return X_train_cost,Y_train_cost,coeff_cost,knn_cost

def test_control_nn(num_data, control_HL1, control_HL2):
    if control_HL2 is not 0:
        model = 'nn/control/NN_control_' + str(num_data) + 'k_' + str(control_HL1) + '_' + str(control_HL2)
    else:
        model = 'nn/control/NN_control_' + str(num_data) + 'k_' + str(control_HL1)
    nn = load_model(model)
    return nn

def test_cost_nn(num_data,cost_HL1,cost_HL2):
    if cost_HL2 is not 0:
        model = 'nn/cost/NN_cost_' + str(num_data) + 'k_' + str(cost_HL1) + '_' + str(cost_HL2)
    else:
        model = 'nn/control/NN_cost_' + str(num_data) + 'k_' + str(cost_HL1)
    nn_cost = load_model(model)
    return nn_cost

def main(flag, num_data = 10, type_data = 'scaled', num_neigh = 5, control_HL1 = 8, control_HL2 = 0, cost_HL1 = 8, cost_HL2 = 0):
    if flag == 0:
        X_train_knn,Y_train_knn,coeff_knn,knn = test_control_knn(num_data,type_data,num_neigh)
        X_train_nn,Y_train_nn,X_validate_cost_nn,Y_validate_cost_nn,coeff_nn = joblib.load('../training_data/control_ft_2_' + str(num_data) + 'k_scaled')
        nn = test_control_nn(num_data,control_HL1,control_HL2)
    else:
        X_train_knn,Y_train_knn,coeff_knn,knn = test_cost_knn(num_data,type_data,num_neigh)
        X_train_nn,Y_train_nn,X_validate_cost_nn,Y_validate_cost_nn,coeff_nn = joblib.load('../training_data/cost_2_' + str(num_data) + 'k_scaled')
        nn = test_control_nn(num_data,cost_HL1,cost_HL2)

    out_knn = knn.predict(X_train_knn[0])
    out_nn = nn.predict(X_train_nn[0].reshape([1,8]))

    print out_knn - Y_train_knn[0]
    print out_nn - Y_train_nn[0]

    print (out_knn*coeff_knn[3] + coeff_knn[2]) - (Y_train_knn[0]*coeff_knn[3] + coeff_knn[2])
    print (out_nn*coeff_nn[3] + coeff_nn[2]) - (Y_train_nn[0]*coeff_nn[3] + coeff_nn[2])

if __name__ == "__main__":
    if len(sys.argv) == 2:
        flag = int(sys.argv[1])
        main(flag)
