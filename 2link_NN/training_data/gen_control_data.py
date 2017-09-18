import sys
import numpy
import preprocess
from sklearn.externals import joblib

def main(flag):
    data = numpy.loadtxt("raw/2_control_data_fulltraj_100k_clean.txt",delimiter="\t")
    # numpy.random.shuffle(data)
    Xi = data[:,0:8] #input
    Yi = data[:,8:48] #output

    if flag == 0:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.scale_data(Xi,Yi)
        file_name = 'control_ft_2_100k_scaled'
        print 'Scaled control data generated'
    elif flag == 1:
        X,Y,X_a,X_b,Y_a,Y_b = preprocess.standardize_data(Xi,Yi)
        file_name = 'control_ft_2_100k_std'
        print 'Standardized control data generated'

    X_train,Y_train,X_validate,Y_validate = preprocess.split_data(X,Y)

    save = (X_train,Y_train, X_validate, Y_validate, [X_a,X_b,Y_a,Y_b])
    joblib.dump(save, file_name)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        flag = int(sys.argv[1])
        main(flag)
