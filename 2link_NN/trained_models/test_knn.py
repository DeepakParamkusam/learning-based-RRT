from sklearn.externals import joblib
import numpy

model = joblib.load('knn_control_ft_2')
test_X=numpy.array([5.68,1.29,-20.42,26.05,0.19,4.32,12.98,-0.54])

Y = model.predict(test_X)
print Y
