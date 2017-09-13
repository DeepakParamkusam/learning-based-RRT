import sys
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

if len(sys.argv) == 3:
    num_data = str(sys.argv[1])
    type_data = str(sys.argv[2])

    data = '../training_data/control_ft_2_' + num_data + 'k_' + type_data
    #load data
    X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)
    #open file to save metrics
    to_file = open('../trained_models/' + type_data + '_knn_control_' + num_data + '.txt', 'a')

    for num_neigh in range(3,16):
        print 'no. of neighbours = ',num_neigh

        #kNN training
        knn = KNeighborsRegressor(n_neighbors=num_neigh)
        knn.fit(X_train, Y_train)

        #validate and compute the mean squared error
        predictions = knn.predict(X_validate)
        mse = mean_squared_error(Y_validate,predictions)
        print "mse =", mse
        to_file.write('%s %s \n' % (num_neigh, mse))

        #save kNN
        joblib.dump(knn, '../trained_models/knn/control/knn_control_ft_2_' + num_data + 'k_' + type_data + '_' + str(num_neigh))
else:
    print 'Incorrect no. of arguments'
