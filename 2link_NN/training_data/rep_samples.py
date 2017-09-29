import numpy
from datetime import datetime

data_cost = numpy.loadtxt('corrected_cleaned_data/2_cost_100k_correct.txt')

states = data_cost[:, 0:8]
cost = data_cost[:, 8]
data_size = len(data_cost)

rep_samples = numpy.array([])
eu_cut_off = 4

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

numpy.savetxt('rep_sample_lists/rep_samples_' + str(eu_cut_off) + '100_4.txt',rep_samples,delimiter='\t')


