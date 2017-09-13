import numpy
from sklearn.externals import joblib

data = numpy.loadtxt("raw/2_control_data_fulltraj_10k_clean.txt",delimiter="\t")
Xi = data[:,0:8] #input
Yi = data[:,8:48] #output

flag = 1

if flag == 1:
    #Scaling
    X = numpy.divide(Xi-Xi.min(axis=0),Xi.max(axis=0)-Xi.min(axis=0))
    Y = numpy.divide(Yi-Yi.min(axis=0),Yi.max(axis=0)-Yi.min(axis=0))
    X_a = Xi.min(axis=0)
    X_b = Xi.max(axis=0)-Xi.min(axis=0)
    Y_a = Yi.min(axis=0)
    Y_b = Yi.max(axis=0)-Yi.min(axis=0)
    file_name = 'control_ft_2_10k_scaled'
    print 'Scaled control data generated'
else:
    #Standardization
    X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
    Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))
    X_a = Xi.mean(axis=0)
    X_b = Xi.std(axis=0)
    Y_a = Yi.mean(axis=0)
    Y_b = Yi.std(axis=0)
    file_name = 'control_ft_2_10k_std'
    print 'Standardized control data generated'

#Split into training data and validation data 9:1
num_data = len(X)
X_train = X[0:int(0.9*num_data),:]
Y_train = Y[0:int(0.9*num_data),:]
X_validate = X[int(0.9*num_data):num_data,:]
Y_validate = Y[int(0.9*num_data):num_data,:]

save = (X_train,Y_train, X_validate, Y_validate, [X_a,X_b,Y_a,Y_b])
joblib.dump(save, file_name)
