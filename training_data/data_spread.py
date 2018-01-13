# Returns mean euclidean distance and its std. deviation of each sample from other samples in the data set
# By Deepak Paramkusam, TU Delft

# Input : Data set as 2-D numpy tuple (standard numpy.loadtxt format)
# Output : Numpy array containing mean distance and its deviation for each sample wrt other samples. Array_size = no. of samples x 2

import numpy
import progressbar

def eudist_mean_dev(states):  
    # Initializing output size  
    data_size = len(states)
    mu_sigma = numpy.zeros([data_size,2])
    
    bar = progressbar.ProgressBar(widgets=[progressbar.SimpleProgress()],max_value=data_size,).start()

    for i in range(0, data_size):       
        # Euclidean dist for each wrt other samples 
        curr_i = numpy.array([[states[i,:]]*data_size])
        diff = states - curr_i
        eu_dist = numpy.linalg.norm(diff,axis=2)[0]

        mu_sigma[i] = [numpy.mean(eu_dist),numpy.std(eu_dist)]
        bar.update(i+1)
    
    bar.finish()
    return mu_sigma
