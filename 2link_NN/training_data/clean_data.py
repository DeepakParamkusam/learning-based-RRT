import numpy

data_control = numpy.loadtxt('raw/control_data_100k.txt')
data_cost = numpy.loadtxt('raw/cost_100k.txt')
data_size = len(data_control)
idx = []
i = 0

while(1):
    if i == data_size:
        break
    bad = any((x > 400.0001 or x < -401.9999) for x in data_control[i][8:48])
    if bad:
        idx.append(i)
    i = i + 1

data_control = numpy.delete(data_control,idx,axis=0)
numpy.savetxt('raw/2_control_data_fulltraj_100k_clean.txt',data_control,delimiter='\t')

data_cost = numpy.delete(data_cost,idx,axis=0)
numpy.savetxt('raw/2_cost_100k_clean.txt',data_cost,delimiter='\t')
