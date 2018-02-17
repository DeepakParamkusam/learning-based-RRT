import numpy
from datetime import datetime

Y = numpy.loadtxt('Y.txt',delimiter=',')
X = numpy.loadtxt('X.txt',delimiter=',')

# cutoff euclidean distance between states 
eu_cut_off = 4
# number of cleaning runs
NUM_RUNS = 100
# number of samples taken into consideration in cleaning run
SAMPLE_RANGE = 500
# SAMPLE_RANGE number of samples are taken from a different point in the dataset to cover larger region.
# Care should therefore be taken such that (NUM_RUNS+1)*SAMPLE_RANGE << initial number of samples 
# The current chosen values are for 100k samples. The code WILL fail for lower sample dataset if the above values are not changed.

for j in range(0,NUM_RUNS):
    print 'run=',j, datetime.now().time()
    rep_samples = numpy.array([])
    cost = Y[:,0]
    states = X[:,0:8]
    data_size = len(states)

    for i in range(j*SAMPLE_RANGE, (j+1)*SAMPLE_RANGE):
        if i in rep_samples:
        	continue
        curr_i = numpy.array([[states[i,:]]*data_size])
        diff = states - curr_i
        eu_dist = numpy.linalg.norm(diff,axis=2)
        
        idx = numpy.where(eu_dist[0] < eu_cut_off)[0]
        min_cost_idx = numpy.where(cost[idx] == numpy.min(cost[idx]))
        idx = numpy.delete(idx,min_cost_idx)
        for each in idx:
        	if each in rep_samples:
        		continue
        	else:
        		rep_samples=numpy.append(rep_samples,each)

    print 'number of samples removed this run=',rep_samples.shape
    Y = numpy.delete(Y,rep_samples,axis=0)
    X = numpy.delete(X,rep_samples,axis=0)
    print X.shape

numpy.savetxt('X_clean.txt',X,delimiter='\t')
numpy.savetxt('Y_clean.txt',Y,delimiter='\t')


