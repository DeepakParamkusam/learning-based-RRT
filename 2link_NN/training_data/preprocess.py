import numpy

def scale_data(Xi,Yi):
    #Scaling
    X = numpy.divide(Xi-Xi.min(axis=0),Xi.max(axis=0)-Xi.min(axis=0))
    Y = numpy.divide(Yi-Yi.min(axis=0),Yi.max(axis=0)-Yi.min(axis=0))
    X_a = Xi.min(axis=0)
    X_b = Xi.max(axis=0)-Xi.min(axis=0)
    Y_a = Yi.min(axis=0)
    Y_b = Yi.max(axis=0)-Yi.min(axis=0)

    return X,Y,X_a,X_b,Y_a,Y_b

def standardize_data(Xi,Yi):
    #Standardization
    X = numpy.divide(Xi-Xi.mean(axis=0),Xi.std(axis=0))
    Y = numpy.divide(Yi-Yi.mean(axis=0),Yi.std(axis=0))
    X_a = Xi.mean(axis=0)
    X_b = Xi.std(axis=0)
    Y_a = Yi.mean(axis=0)
    Y_b = Yi.std(axis=0)

    return X,Y,X_a,X_b,Y_a,Y_b

def log_scale_data(Xi,Yi):
    Yi = numpy.log(Yi)
    X,Y,X_a,X_b,Y_a,Y_b = scale_data(Xi,Yi)

    return X,Y,X_a,X_b,Y_a,Y_b

def split_data(X,Y):
    #Split into training data and validation data 9:1
    num_data = len(X)
    X_train = X[0:int(0.9*num_data),:]
    Y_train = Y[0:int(0.9*num_data),:]
    X_validate = X[int(0.9*num_data):num_data,:]
    Y_validate = Y[int(0.9*num_data):num_data,:]
    return X_train,Y_train,X_validate,Y_validate
