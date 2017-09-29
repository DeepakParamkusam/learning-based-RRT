import sys
import numpy
import preprocess
from sklearn.externals import joblib

def main(num_data,rep_parameter,flag):
    data = numpy.loadtxt('corrected_clean_data/2_cost_' + str(num_data) + 'k_clean_' + str(rep_parameter) + '.txt',delimiter="\t")
    Xi = data[:,0:8] #input
    Yi = data[:,8:9] #output

    if flag == 0:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.scale_data(Xi,Yi)
        file_name = 'final_data/final_cost_2_' + str(num_data) + 'k_scaled_' + str(rep_parameter)
        print 'Scaled cost data generated'
    elif flag == 1:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.standardize_data(Xi,Yi)
        file_name = 'final_data/final_cost_2_' + str(num_data) + 'k_std_' + str(rep_parameter)
        print 'Standardized cost data generated'
    elif flag == 2:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.log_scale_data(Xi,Yi)
        file_name = 'final_data/final_cost_2_' + str(num_data) + 'k_scaled_log_' + str(rep_parameter)
        print 'Log scaled cost data generated'

    X_train,Y_train,X_validate,Y_validate = preprocess.split_data(X,Y)

    save = (X_train,Y_train, X_validate, Y_validate, [X_a,X_b,Y_a,Y_b])
    joblib.dump(save, file_name)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
