import numpy
from datetime import datetime

data_cost = numpy.loadtxt('clean_indirect/Y100k.txt',delimiter=',',usecols=0)
data_control = numpy.loadtxt('clean_indirect/Y100k.txt',delimiter=',',usecols=(2,3,4,5))
states = numpy.loadtxt('clean_indirect/X100k.txt',delimiter=',')

eu_cut_off = 4
NUM_RUNS = 100
SAMPLE_RANGE = 500

for j in range(0,NUM_RUNS):
    print 'run=',j, datetime.now().time()
    rep_samples = numpy.array([])
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
    data_control = numpy.delete(data_control,rep_samples,axis=0)
    data_cost = numpy.delete(data_cost,rep_samples,axis=0)
    print data_control.shape

numpy.savetxt('clean_direct/control_500k_clean.txt',data_control,delimiter='\t')
numpy.savetxt('clean_direct/cost_500k_clean.txt',data_cost,delimiter='\t')


