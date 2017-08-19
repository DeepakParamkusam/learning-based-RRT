import numpy as np
import rrt
import plant

control = np.loadtxt('control2.txt')

STATE_RANGE = np.array([[-3/2*np.pi,-3/2*np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi,np.pi]])
COSTATE_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01]])
PHI_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[3*np.pi/2-0.01,3*np.pi/2-0.01,3*np.pi/2-0.01]])

iState = np.zeros(4)
fState = np.zeros(4)
fState[0] = 2*np.pi

iState[0] = 3.05175638198853
iState[1] =	2.35711622238159
iState[2] = -0.212169647216797
iState[3] = -11.7048187255859

# testNode = rrt.TreeNode()
modelP = rrt.modelPredict(iState,fState)
# gr = rrt.goalReached(iState,fState)
# samp_node = rrt.sampleState(4,STATE_RANGE,fState,0.15)
#
# c = np.hstack((np.vstack((samp_node,samp_node)),np.vstack((iState,fState))))
# neighP = rrt.neighPredict(c)
# print control[0,:]
# a = rrt.connectNodes(samp_node,fState[0])
# a = plant.RK4Simulate(iState,control)
print modelP[1][0][0]
