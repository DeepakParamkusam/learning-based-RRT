import numpy as np

def plantRRTSimulate(costFinal,q0,phi0):
	combinedState0 = plantInitialState(q0,phi0)
	if costFinal == 0:
		output = q0
	else:
		# TODO: Magic number 201: Split simulation into 201 steps?
		# Depends on the valid simulation time and dt
		dcost = costFinal/101
		simulationResult = RK4Simulate(plantEOMoverCost,costFinal,combinedState0[0:-1],dcost)
		output = simulationResult[1][-1,0:q0.size]
	return output

def plantEOMoverCost(cost,combinedState):
	fullState = np.hstack((combinedState,cost))
	fullStateDot = plantEOM(0,fullState)
	combinedStateDot = fullStateDot[0:-1]/fullStateDot[-1]
	return combinedStateDot;

def RK4Step(eom,time,state,dt):
	k1 = eom(time,state)
	k2 = eom(time + dt/2, state + dt/2*k1)
	k3 = eom(time + dt/2, state + dt/2*k2)
	k4 = eom(time + dt, state + dt*k3)
	stateNew = state + dt/6*(k1+2*k2+2*k3+k4)
	
	return stateNew;

def RK4Simulate(eom,tFinal,state0,dt):
	numberOfSteps = tFinal/dt + 1
	time = np.linspace(0,tFinal,numberOfSteps).ravel()
	state = np.zeros((time.shape[0],state0.shape[0]))
	state[0,:] = state0
	for count in np.arange(0,time.shape[0]-1):
		state[count+1,:] = RK4Step(eom,time[count],state[count,:],dt)
	output = [time,state]
	return output;

def plantEOM(time,fullState):
	q1 = fullState[0]
	q2 = fullState[1]
	qd1 = fullState[2]
	qd2 = fullState[3]
	lbd1 = fullState[4]
	lbd2 = fullState[5]
	lbd3 = fullState[6]
	lbd4 = fullState[7]

	q1Dot = qd1
	q2Dot = qd2
	qd1Dot = -(900*lbd3 - 2376*lbd4 - np.sin(q2)*(330*qd1**2 + 660*qd1*qd2 + 330*qd2**2) - 207*qd1**2*np.sin(2*q2) + 54*qd1**2*np.sin(3*q2) + (81*qd1**2*np.sin(4*q2))/2 + 54*qd2**2*np.sin(3*q2) + np.cos(2*q2)*(324*lbd3 - 648*lbd4) + 864*lbd3*np.cos(q2) - 3456*lbd4*np.cos(q2) + 108*qd1*qd2*np.sin(3*q2))/(9*np.cos(2*q2) - 23)**2
	qd2Dot = (2376*lbd3 - 9108*lbd4 - np.sin(q2)*(1650*qd1**2 + 660*qd1*qd2 + 330*qd2**2) - 414*qd1**2*np.sin(2*q2) + 270*qd1**2*np.sin(3*q2) - 207*qd2**2*np.sin(2*q2) + 81*qd1**2*np.sin(4*q2) + 54*qd2**2*np.sin(3*q2) + (81*qd2**2*np.sin(4*q2))/2 + np.cos(2*q2)*(648*lbd3 - 1620*lbd4) + 3456*lbd3*np.cos(q2) - 9504*lbd4*np.cos(q2) - 414*qd1*qd2*np.sin(2*q2) + 108*qd1*qd2*np.sin(3*q2) + 81*qd1*qd2*np.sin(4*q2))/(9*np.cos(2*q2) - 23)**2

	lbd1Dot = 0
	lbd2Dot = (19656*lbd3**2*np.sin(q2) + 216216*lbd4**2*np.sin(q2) + 23652*lbd3**2*np.sin(2*q2) + 5832*lbd3**2*np.sin(3*q2) + 201204*lbd4**2*np.sin(2*q2) + 1458*lbd3**2*np.sin(4*q2) + 64152*lbd4**2*np.sin(3*q2) + 7290*lbd4**2*np.sin(4*q2) - np.cos(2*q2)*(21960*lbd4*qd1**2 - 10980*lbd3*qd1**2 + 10980*lbd4*qd2**2 + 21960*lbd4*qd1*qd2) - 5589*lbd3*qd1**2 + 11178*lbd4*qd1**2 + 5589*lbd4*qd2**2 - 157248*lbd3*lbd4*np.sin(q2) + 1866*lbd3*qd1**2*np.cos(q2) + 1866*lbd3*qd2**2*np.cos(q2) - 9330*lbd4*qd1**2*np.cos(q2) - 1866*lbd4*qd2**2*np.cos(q2) - 115344*lbd3*lbd4*np.sin(2*q2) - 46656*lbd3*lbd4*np.sin(3*q2) - 5832*lbd3*lbd4*np.sin(4*q2) + 11178*lbd4*qd1*qd2 + 729*lbd3*qd1**2*np.cos(3*q2) - 1863*lbd3*qd1**2*np.cos(4*q2) + 729*lbd3*qd2**2*np.cos(3*q2) - 3645*lbd4*qd1**2*np.cos(3*q2) - 243*lbd3*qd1**2*np.cos(5*q2) + 3726*lbd4*qd1**2*np.cos(4*q2) - 729*lbd4*qd2**2*np.cos(3*q2) - 243*lbd3*qd2**2*np.cos(5*q2) + 1215*lbd4*qd1**2*np.cos(5*q2) + 1863*lbd4*qd2**2*np.cos(4*q2) + 243*lbd4*qd2**2*np.cos(5*q2) + 3732*lbd3*qd1*qd2*np.cos(q2) - 3732*lbd4*qd1*qd2*np.cos(q2) + 1458*lbd3*qd1*qd2*np.cos(3*q2) - 1458*lbd4*qd1*qd2*np.cos(3*q2) - 486*lbd3*qd1*qd2*np.cos(5*q2) + 3726*lbd4*qd1*qd2*np.cos(4*q2) + 486*lbd4*qd1*qd2*np.cos(5*q2))/(9*np.cos(2*q2) - 23)**3
	lbd3Dot = - lbd1 - (np.sin(q2)*(12*lbd3*qd1 + 12*lbd3*qd2 - 60*lbd4*qd1 - 12*lbd4*qd2) + 9*lbd3*qd1*np.sin(2*q2) - 18*lbd4*qd1*np.sin(2*q2) - 9*lbd4*qd2*np.sin(2*q2))/(9*np.sin(q2)**2 + 7)
	lbd4Dot = (6*lbd4*(qd1 + qd2)*((3*np.sin(2*q2))/2 + 2*np.sin(q2)))/(9*np.sin(q2)**2 + 7) - (3*lbd3*np.sin(q2)*(4*qd1 + 4*qd2))/(9*np.sin(q2)**2 + 7) - lbd2
	
	costDot = (6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4))**2/(9*np.cos(q2)**2 - 16)**2 + (9*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2))**2)/(9*np.cos(q2)**2 - 16)**2

	fullStateDot = np.array([q1Dot,q2Dot,qd1Dot,qd2Dot,lbd1Dot,lbd2Dot,lbd3Dot,lbd4Dot,costDot])
	return fullStateDot

def plantInitialState(x0,phi0):
	q1 = x0[0]
	q2 = x0[1]
	qd1 = x0[2]
	qd2 = x0[3]

	lbd1 = phi0[0]
	lbd2 = phi0[1]
	lbd3 = phi0[2]
	
	lbd4 = -((3*np.cos(q2) - 4)*(3*np.cos(q2) + 4)*(288*lbd3 - 320*qd1**2*np.sin(q2) - 64*qd2**2*np.sin(q2) - 23*(416*lbd1*qd1 + 416*lbd2*qd2 + 100*qd1**4*np.sin(q2)**2 + 4*qd2**4*np.sin(q2)**2 - 36*lbd3**2 + 528*lbd1*qd1*np.cos(q2) + 528*lbd2*qd2*np.cos(q2) + 56*qd1**2*qd2**2*np.sin(q2)**2 + 120*qd1**4*np.cos(q2)*np.sin(q2)**2 + 12*qd2**4*np.cos(q2)*np.sin(q2)**2 + 180*lbd1*qd1*np.cos(q2)**2 + 180*lbd2*qd2*np.cos(q2)**2 - 24*lbd3*qd1**2*np.sin(q2) + 120*lbd3*qd2**2*np.sin(q2) + 36*qd1**4*np.cos(q2)**2*np.sin(q2)**2 + 9*qd2**4*np.cos(q2)**2*np.sin(q2)**2 + 16*qd1*qd2**3*np.sin(q2)**2 + 80*qd1**3*qd2*np.sin(q2)**2 + 132*qd1**2*qd2**2*np.cos(q2)*np.sin(q2)**2 + 36*qd1*qd2**3*np.cos(q2)**2*np.sin(q2)**2 + 72*qd1**3*qd2*np.cos(q2)**2*np.sin(q2)**2 - 36*lbd3*qd1**2*np.cos(q2)*np.sin(q2) + 72*lbd3*qd2**2*np.cos(q2)*np.sin(q2) + 72*qd1**2*qd2**2*np.cos(q2)**2*np.sin(q2)**2 + 240*lbd3*qd1*qd2*np.sin(q2) + 48*qd1*qd2**3*np.cos(q2)*np.sin(q2)**2 + 168*qd1**3*qd2*np.cos(q2)*np.sin(q2)**2 + 144*lbd3*qd1*qd2*np.cos(q2)*np.sin(q2))**(1/2) - 96*qd1**2*np.sin(2*q2) - 48*qd2**2*np.sin(2*q2) + 576*lbd3*np.cos(q2) + 9*np.cos(2*q2)*(416*lbd1*qd1 + 416*lbd2*qd2 + 100*qd1**4*np.sin(q2)**2 + 4*qd2**4*np.sin(q2)**2 - 36*lbd3**2 + 528*lbd1*qd1*np.cos(q2) + 528*lbd2*qd2*np.cos(q2) + 56*qd1**2*qd2**2*np.sin(q2)**2 + 120*qd1**4*np.cos(q2)*np.sin(q2)**2 + 12*qd2**4*np.cos(q2)*np.sin(q2)**2 + 180*lbd1*qd1*np.cos(q2)**2 + 180*lbd2*qd2*np.cos(q2)**2 - 24*lbd3*qd1**2*np.sin(q2) + 120*lbd3*qd2**2*np.sin(q2) + 36*qd1**4*np.cos(q2)**2*np.sin(q2)**2 + 9*qd2**4*np.cos(q2)**2*np.sin(q2)**2 + 16*qd1*qd2**3*np.sin(q2)**2 + 80*qd1**3*qd2*np.sin(q2)**2 + 132*qd1**2*qd2**2*np.cos(q2)*np.sin(q2)**2 + 36*qd1*qd2**3*np.cos(q2)**2*np.sin(q2)**2 + 72*qd1**3*qd2*np.cos(q2)**2*np.sin(q2)**2 - 36*lbd3*qd1**2*np.cos(q2)*np.sin(q2) + 72*lbd3*qd2**2*np.cos(q2)*np.sin(q2) + 72*qd1**2*qd2**2*np.cos(q2)**2*np.sin(q2)**2 + 240*lbd3*qd1*qd2*np.sin(q2) + 48*qd1*qd2**3*np.cos(q2)*np.sin(q2)**2 + 168*qd1**3*qd2*np.cos(q2)*np.sin(q2)**2 + 144*lbd3*qd1*qd2*np.cos(q2)*np.sin(q2))**(1/2) + 216*lbd3*np.cos(q2)**2 - 128*qd1*qd2*np.sin(q2) + 180*qd1**2*np.cos(q2)**2*np.sin(q2) + 36*qd2**2*np.cos(q2)**2*np.sin(q2) - 96*qd1*qd2*np.sin(2*q2) + 54*qd1**2*np.sin(2*q2)*np.cos(q2)**2 + 27*qd2**2*np.sin(2*q2)*np.cos(q2)**2 + 72*qd1*qd2*np.cos(q2)**2*np.sin(q2) + 54*qd1*qd2*np.sin(2*q2)*np.cos(q2)**2))/(6*(936*np.cos(2*q2) + 5412*np.cos(q2) - 1899*np.cos(q2)**2 - 4752*np.cos(q2)**3 - 1620*np.cos(q2)**4 + 1188*np.cos(2*q2)*np.cos(q2) + 405*np.cos(2*q2)*np.cos(q2)**2 + 4264))
	cost = 0
	if np.isnan(lbd4):
		lbd4 = 0
		cost = 50

	if np.sign(np.cos(q2)) > 0:
		lbd4= -lbd4
	x0 = np.array([q1,q2,qd1,qd2,lbd1,lbd2,lbd3,lbd4,cost])
	return x0