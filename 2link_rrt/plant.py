import numpy as np

def eom(t,state,control):
	q1=state[0]
	q2=state[1]
	qd1=state[2]
	qd2=state[3]
	tau1=control[0]
	tau2=control[1]

	qdd1 = -(48*tau1 - 48*tau2 + 24*qd1*qd1*np.sin(q2) + 24*qd2*qd2*np.sin(q2) + 18*qd1*qd1*np.sin(2*q2) - 72*tau2*np.cos(q2) + 48*qd1*qd2*np.sin(q2))/(36*np.cos(q2)*np.cos(q2) - 64)
	qdd2 = (48*tau1 - 240*tau2 + 120*qd1*qd1*np.sin(q2) + 24*qd2*qd2*np.sin(q2) + 36*qd1*qd1*np.sin(2*q2) + 18*qd2*qd2*np.sin(2*q2) + 72*tau1*np.cos(q2) - 144*tau2*np.cos(q2) + 48*qd1*qd2*np.sin(q2) + 36*qd1*qd2*np.sin(2*q2))/(18*np.cos(2*q2) - 46)

	stateDot = np.array([qd1,qd2,qdd1,qdd2])
	return stateDot

def RK4Step(time,state,dt,control):
	k1 = eom(time,state,control)
	k2 = eom(time + dt/2, state + dt/2*k1,control)
	k3 = eom(time + dt/2, state + dt/2*k2,control)
	k4 = eom(time + dt, state + dt*k3,control)
	stateNew = state + dt/6*(k1+2*k2+2*k3+k4)

def RK4Simulate(state0,control):
	numberOfSteps = 21
	totalTime = 0.5
	dt = totalTime/numberOfSteps
	time = np.linspace(0,totalTime,numberOfSteps).ravel()
	state = np.zeros((time.shape[0],state0.shape[0]))
	state[0,:] = state0
	for count in np.arange(0,time.shape[0]-1):
		state[count+1,:] = RK4Step(time[count],state[count,:],dt,control[count,:])
	output = [time,state]
	return output;
