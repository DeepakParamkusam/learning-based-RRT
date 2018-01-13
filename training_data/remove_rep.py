import numpy 

euc_dist =2

rep_samp = numpy.loadtxt('indirect_repsamples/indirec_rep_samples_unconstrained_'+str(euc_dist)+'.txt',delimiter=',')
data_control = numpy.loadtxt('raw_indirect/Y100k.txt',delimiter=',')
data_cost = numpy.loadtxt('raw_indirect/X100k.txt',delimiter=',')

control_new = numpy.delete(data_control[0:50000,:],rep_samp,axis=0)
cost_new = numpy.delete(data_cost[0:50000,:],rep_samp,axis=0)
print len(rep_samp),'samples removed'

numpy.savetxt('indirect_unconstrained_clean'+str(euc_dist)+'_control.txt',control_new,delimiter='\t')
numpy.savetxt('indirect_unconstrained_clean'+str(euc_dist)+'_states.txt',cost_new,delimiter='\t')