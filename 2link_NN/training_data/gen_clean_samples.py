import sys
import numpy

def main(data_size,rep_parameter):
	rep_samp = numpy.loadtxt('rep_sample_lists/rep_samples_' + str(data_size) + '_' + str(rep_parameter) + '.txt')
	data_control = numpy.loadtxt('corrected_clean_data/2_control_data_fulltraj_' + str(data_size) + 'k_correct.txt')
	data_cost = numpy.loadtxt('corrected_clean_data/2_cost_' + str(data_size) + 'k_correct.txt')

	control_new = numpy.delete(data_control,rep_samp,axis=0)
	cost_new = numpy.delete(data_cost,rep_samp,axis=0)
	print len(rep_samp),'samples removed'

	numpy.savetxt('corrected_clean_data/2_control_data_fulltraj_' + str(data_size) + 'k_clean_' + str(rep_parameter) + '.txt',control_new,delimiter='\t')
	numpy.savetxt('corrected_clean_data/2_cost_' + str(data_size) + 'k_clean_' + str(rep_parameter) + '.txt',cost_new,delimiter='\t')

if __name__ == "__main__":
	if len(sys.argv) == 3:
		main(sys.argv[1],sys.argv[2])

