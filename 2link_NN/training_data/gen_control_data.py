import sys
import numpy
import preprocess
from sklearn.externals import joblib

def main(num_data,rep_parameter,flag):
    data = numpy.loadtxt('corrected_clean_data/2_control_data_fulltraj_' + str(num_data) + 'k_clean_' + str(rep_parameter) + '.txt',delimiter="\t")
    Xi = data[:,0:8] #input
    Yi = data[:,8:48] #output

    if flag == 0:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.scale_data(Xi,Yi)
        file_name = 'final_data/final_control_ft_2_' + str(num_data) + 'k_scaled_' + str(rep_parameter)
        print 'Scaled control data generated'
    elif flag == 1:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.standardize_data(Xi,Yi)
        file_name = 'final_data/final_control_ft_2_' + str(num_data) + 'k_std_' + str(rep_parameter)
        print 'Standardized control data generated'

    X_train,Y_train,X_validate,Y_validate = preprocess.split_data(X,Y)

    save = (X_train,Y_train, X_validate, Y_validate, [X_a,X_b,Y_a,Y_b])
    joblib.dump(save, file_name)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
