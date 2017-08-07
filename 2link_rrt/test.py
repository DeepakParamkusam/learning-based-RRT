import numpy as np
import rrt
import plant

control = np.loadtxt('control.txt')

STATE_RANGE = np.array([[-3/2*np.pi,-3/2*np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi,np.pi]])
COSTATE_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01]])
PHI_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[3*np.pi/2-0.01,3*np.pi/2-0.01,3*np.pi/2-0.01]])

iState = np.zeros(4)
# fState = np.zeros((1,4))
# fState[0] = 2*np.pi
iState[0] = 5.68
iState[1] =	1.29
iState[2] = -20.42
iState[3] = 26.05

# testNode = rrt.TreeNode()
# modelP = rrt.modelPredict(iState,fState)
# gr = rrt.goalReached(iState,fState)
# samp_node = rrt.sampleState(4,STATE_RANGE,fState,0.15)
#
# c = np.hstack((np.vstack((samp_node,samp_node)),np.vstack((iState,fState))))
# neighP = rrt.neighPredict(c)

# a = rrt.connectNodes(samp_node,fState[0])
a = plant.RK4Simulate(iState,control)
print a
