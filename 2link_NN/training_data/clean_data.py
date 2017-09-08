import numpy

data = numpy.loadtxt('raw/2_control_data_fulltraj_10k.txt')
data_size = len(data)
sim_data = []

for idx in range(0,data_size-1):
    if idx%1000 == 0:
        print idx
    for jdx in range(idx+1,data_size):
        if numpy.linalg.norm(data[idx][0:8] - data[jdx][0:8]) < 1:
            sim_data = [sim_data,idx]

print sim_data
