import sys
import numpy
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from numpy.random import seed

seed(1)

if len(sys.argv) == 4:
    num_data = str(sys.argv[1])
    type_data = str(sys.argv[2])
    rep_parameter = str(sys.argv[3])

    data = '../training_data/final_data/final_cost_2_' + num_data + 'k_' + type_data + '_' + rep_parameter
    #load data
    X_train,Y_train,X_validate,Y_validate,coeff = joblib.load(data)
    #open file to save metrics
    to_file = open('../trained_models/' + type_data + '_knn_cost_' + num_data + '_' + rep_parameter + '.txt', 'a')

    for num_neigh in range(5,10):
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
        joblib.dump(knn, '../trained_models/knn/cost/knn_cost_2_' + num_data + 'k_' + type_data + '_' + rep_parameter + '_' + str(num_neigh))
else:
    print 'Incorrect no. of arguments'
