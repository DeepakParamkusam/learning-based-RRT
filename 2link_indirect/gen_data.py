import numpy as np
from scipy.optimize import fsolve

class PlantParameters():
	NO_OF_STATES = 4
	STATE_RANGE = np.array([[0,0,-30,-30],[2*np.pi,2*np.pi,30,30]])
	COSTATE_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01]])
	PHI_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[3*np.pi/2-0.01,3*np.pi/2-0.01,3*np.pi/2-0.01]])

	SIM_T0 = 0.
	SIM_T1 = 0.5
	SIM_DT = 0.025

	BOUND_TIME = 0.5
	BOUND_STATE_DIFF = 1.5
	BOUND_COSTATE = [np.inf,np.inf,np.inf,np.inf]
	BOUND_COST = 2000

def plantDataRandom(numberOfSimulations,tFinal,dt,subSample,stateRange,phiRange,tBound,stateBound,costBound,costateBound,controlConstrained):
	# Compute number of samples used
	samplesPerSimulation = (np.arange(0,tFinal+dt,dt)[subSample-1::subSample]).shape[0]
	# Find the number of states
	nStates = stateRange.shape[1]
	Xdata = np.zeros((samplesPerSimulation*numberOfSimulations,2*nStates))
	# If so, what is stored in Ydata?
	Ydata = np.zeros((samplesPerSimulation*numberOfSimulations,2+nStates))
	# Do $numberOfSimulations runs
	for count in np.arange(0,numberOfSimulations):
		print count
		# Generate random starting state in state space
		x0 = np.random.uniform(stateRange[0], 	stateRange[1])
		# Keep searching for a valid starting point
		foundValidStartingPoint = False
		noPointFoundCount = 0
		while not(foundValidStartingPoint):
			noPointFoundCount += 1
			# Take a random costate
			phi0 = np.random.uniform(phiRange[0], phiRange[1])
			# Compute the full initial state
			fullState0 = plantInitialState(x0, phi0)
			# Find whether the initial state is valid
			isValid = plantIsValid(0,tBound,x0,x0,stateBound,
									fullState0[-1],costBound,
									np.array([fullState0[nStates:-1]]),costateBound,controlConstrained)

			if isValid == 1:
				foundValidStartingPoint = True
			if noPointFoundCount > 10:
				x0 = np.random.uniform(stateRange[0], 	stateRange[1])

		# Simulate the model
		XdataRun, YdataRun = plantDataRun(tFinal,dt,subSample,x0,phi0,tBound,stateBound,costBound,costateBound,phiRange,controlConstrained)

		# Store the data from this run in Xdata and Ydata
		Xdata[(count)*samplesPerSimulation:((count+1)*samplesPerSimulation),:] = XdataRun
		Ydata[(count)*samplesPerSimulation:((count+1)*samplesPerSimulation),:] = YdataRun

	# All runs done
	# Select only valid data
	# print Ydata[:,1]
	print np.sum(Ydata[:,1]==2),"samples removed"
	Xdata = Xdata[Ydata[:,1]<2,:]
	Ydata = Ydata[Ydata[:,1]<2,:]
	# Return
	return[Xdata,Ydata];

def plantDataRun(tFinal,dtSimulate,subSample,x0,phi0,tBound,stateBound,costBound,costateBound,phiRange,controlConstrained):
	# simulate model
	fullState0 = plantInitialState(x0,phi0)
	simulationResult = RK4Simulate(plantEOM,tFinal,fullState0,dtSimulate)
	timeResult = simulationResult[0]
	fullStateResult = simulationResult[1]

	# sample results
	timeSamples = timeResult[subSample-1::subSample]
	fullStateSamples = fullStateResult[subSample-1::subSample,:]
	nStates = x0.size
	stateSamples = fullStateSamples[:,0:nStates]
	costateSamples = fullStateSamples[:,nStates:-1]
	costSamples = fullStateSamples[:,2*nStates]

	# compute additional in- and output value
	state0Samples = np.tile(x0,(timeSamples.shape[0],1))
	costate0Samples = np.tile(fullState0[nStates:-1],(timeSamples.shape[0],1))
	phi0Samples = costate0Samples[0,:] * np.ones((timeSamples.shape[0],nStates))
	isValidSamples = plantIsValid(timeSamples,tBound,stateSamples,x0,stateBound,costSamples,costBound,costateSamples,costateBound,controlConstrained)
	# collect results
	Xoutput = np.hstack((state0Samples,stateSamples))
	Youtput = np.zeros((timeSamples.shape[0],2+nStates))
	Youtput[:,0] = np.transpose(costSamples)
	Youtput[:,1] = np.transpose(isValidSamples)
	Youtput[:,2:] = phi0Samples

	print phi0Samples

	return[Xoutput,Youtput];

def plantIsValid(time,tBound,q,q0,stateBound,cost,costBound,costate,costateBound,controlConstrained):
	if len(q.shape) == 1:
		q = np.reshape(q,(1,-1))
	qDifference = q-q0
	if len(qDifference.shape) == 1:
		qDifference = np.reshape(qDifference,(1,qDifference.size))
	if len(cost.shape) == 0:
		cost = np.array([[cost]])
	cost = np.reshape(cost,cost.shape[0])
	stateNorm = np.linalg.norm(qDifference,axis=1)
	coStateValid = np.abs(costate) < costateBound
	coStateValid = np.sum(coStateValid,axis=1) == q.shape[1]
	coStateValid = booleanAllBefore(coStateValid)

	# Check if time, states and cost are within bounds
	isValidTrue = np.logical_and((time < tBound), (stateNorm < stateBound))
	isValidTrue = np.logical_and(isValidTrue,(cost < costBound))
	isValidTrue = np.logical_and(isValidTrue,coStateValid)

	isValidFalse = np.logical_and((time < 1.5*tBound),(stateNorm < 1.5*stateBound))
	isValidFalse = np.logical_and(isValidFalse,(cost <1.5 *costBound))
	isValidFalse = np.logical_and(isValidFalse, coStateValid)

	#check control contraint
	if controlConstrained == True:
		controlValid = checkControlConstraint(q,costate)
		isValidTrue = np.logical_and(isValidTrue,controlValid)
		isValidFalse = np.logical_and(isValidFalse,controlValid)

	isValid = 2*np.ones(isValidTrue.shape)
	isValid[isValidFalse] = 0
	isValid[isValidTrue] = 1

	return isValid

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

	Hstar = lambda lbd4 : (6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4))**2/(9*np.cos(q2)**2 - 16)**2 + lbd1*qd1 + lbd2*qd2 + (9*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2))**2)/(9*np.cos(q2)**2 - 16)**2 + (6*lbd4*(10*qd1**2*np.sin(q2) + 2*qd2**2*np.sin(q2) - (12*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + 3*qd1**2*np.sin(2*q2) + (3*qd2**2*np.sin(2*q2))/2 + (20*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) - (18*np.cos(q2)*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + 4*qd1*qd2*np.sin(q2) + 3*qd1*qd2*np.sin(2*q2) + (12*np.cos(q2)*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16)))/(9*np.cos(2*q2) - 23) - (3*lbd3*(2*qd1**2*np.sin(q2) + 2*qd2**2*np.sin(q2) - (12*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + (4*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) + 4*qd1*qd2*np.sin(q2) + (6*np.cos(q2)*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) + 3*qd1**2*np.cos(q2)*np.sin(q2)))/(9*np.cos(q2)**2 - 16)

	lbd4_initial_guess = np.random.uniform(-np.pi,np.pi,1)
	lbd4 = fsolve(Hstar, lbd4_initial_guess)

	#print lbd4

	lbd4 = lbd4[0]
	Hstar_value = (6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4))**2/(9*np.cos(q2)**2 - 16)**2 + lbd1*qd1 + lbd2*qd2 + (9*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2))**2)/(9*np.cos(q2)**2 - 16)**2 + (6*lbd4*(10*qd1**2*np.sin(q2) + 2*qd2**2*np.sin(q2) - (12*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + 3*qd1**2*np.sin(2*q2) + (3*qd2**2*np.sin(2*q2))/2 + (20*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) - (18*np.cos(q2)*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + 4*qd1*qd2*np.sin(q2) + 3*qd1*qd2*np.sin(2*q2) + (12*np.cos(q2)*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16)))/(9*np.cos(2*q2) - 23) - (3*lbd3*(2*qd1**2*np.sin(q2) + 2*qd2**2*np.sin(q2) - (12*(2*lbd4 - 2*lbd3 + 3*lbd4*np.cos(q2)))/(9*np.cos(q2)**2 - 16) + (4*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) + 4*qd1*qd2*np.sin(q2) + (6*np.cos(q2)*(6*lbd3 - 30*lbd4 + np.cos(q2)*(9*lbd3 - 18*lbd4)))/(9*np.cos(q2)**2 - 16) + 3*qd1**2*np.cos(q2)*np.sin(q2)))/(9*np.cos(q2)**2 - 16)

	#print("Hstar_value:", Hstar_value)

	cost = 0
	# if np.isnan(lbd4):
	# 	lbd4 = 0
	# 	cost = 50
	#
	# if np.sign(np.cos(q2)) > 0:
	# 	lbd4= -lbd4
	x0 = np.array([q1,q2,qd1,qd2,lbd1,lbd2,lbd3,lbd4,cost])
	return x0

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

def booleanAllBefore(listOfBools):
	newList = listOfBools
	for c1 in np.arange(1,len(listOfBools)):
		newList[c1] = newList[c1-1] and newList[c1]
	return newList;

def checkControlConstraint(x,phi):
	stateLen = x.shape[0]
	controlValid = np.zeros((stateLen,),dtype=bool)
	for i in range(0,stateLen):
		u1 = -(3*(2*phi[i][3] - 2*phi[i][2] + 3*phi[i][3]*np.cos(x[i][1])))/(9*np.cos(x[i][1])**2 - 16)
		u2 = -(6*phi[i][2] - 30*phi[i][3] + np.cos(x[i][1])*(9*phi[i][2] - 18*phi[i][3]))/(9*np.cos(x[i][1])**2 - 16)

		#hard coded control limit
		if (np.abs(u1)<400) or (np.abs(u2)<400):
			controlValid[i] = True
		# else:
			# print "CONSTRAINT VIOLATED!!"
	return controlValid
