import numpy as np
from gen_data import plantDataRandom
from gen_data import PlantParameters
from datetime import datetime


t0 = PlantParameters().SIM_T0
t1 = PlantParameters().SIM_T1
dt = PlantParameters().SIM_DT

stateRange = PlantParameters().STATE_RANGE
costateRange = PlantParameters().COSTATE_RANGE
phiRange = PlantParameters().PHI_RANGE

subSample = 1
numberOfSimulations = 100000

tBound = PlantParameters().BOUND_TIME
stateBound = PlantParameters().BOUND_STATE_DIFF
costBound = PlantParameters().BOUND_COST
costateBound = PlantParameters().BOUND_COSTATE
controlConstrained = True

print "Generating data samples..."
np.random.seed(42)
init_time = str(datetime.now())
Xdata,Ydata = plantDataRandom(numberOfSimulations,t1,dt,subSample,stateRange,phiRange,tBound,stateBound, costBound,costateBound,controlConstrained)
end_time = str(datetime.now())

XdataValid = Xdata
YdataValid = Ydata

np.savetxt('code_check_test_x_cons.txt', XdataValid, delimiter=',')
np.savetxt('code_check_test_y_cons.txt', YdataValid, delimiter=',')

print 'start_time=',init_time
print 'end_time=',end_time