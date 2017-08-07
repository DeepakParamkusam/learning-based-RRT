import numpy
from sklearn import neighbors as nb
from sklearn.externals import joblib

data = numpy.loadtxt("../training_data/2_cost.txt",delimiter="\t")
Xi = data[:,0:8] #input

neigh_model = nb.NearestNeighbors(5)
neigh_model.fit(Xi)

joblib.dump(neigh_model, '../trained_models/knn_neigh_2')
