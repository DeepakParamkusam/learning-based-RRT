import numpy

data_control = numpy.loadtxt('raw/control_data_100k.txt')
data_cost = numpy.loadtxt('raw/cost_100k.txt')
data_size = len(data_control)
idx = []
i = 0

x_max = 400.0001
x_min = -401.9999
# x_max = 12000.0001
# x_min = -12001.9999

while(1):
    if i == data_size:
        break
    bad = any((x > x_max or x < x_min) for x in data_control[i][8:48])
    if bad:
        idx.append(i)
    i = i + 1

data_control = numpy.delete(data_control,idx,axis=0)
numpy.savetxt('corrected_clean_data/2_control_data_fulltraj_100k_correct.txt',data_control,delimiter='\t')

data_cost = numpy.delete(data_cost,idx,axis=0)
numpy.savetxt('corrected_clean_data/2_cost_100k_correct.txt',data_cost,delimiter='\t')
