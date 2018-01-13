import numpy
from datetime import datetime

# data = numpy.loadtxt('raw/cost_100k.txt')
# states = data[0:50000,0:8]
# cost = data[0:50000,8]
cost = numpy.loadtxt('raw_indirect/Y100k.txt',delimiter=',',usecols=(0,))
states = numpy.loadtxt('raw_indirect/X100k.txt',delimiter=',')

cost=cost[0:50000]
states=states[0:50000,:]
data_size = len(states)

rep_samples = numpy.array([])
eu_cut_off = 2

for i in range(0, data_size):
    if i%1000 == 0:
    	print i/1000, datetime.now().time()
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

numpy.savetxt('indirect_repsamples/indirec_rep_samples_unconstrained_'+ str(eu_cut_off) + '.txt',rep_samples,delimiter='\t')


