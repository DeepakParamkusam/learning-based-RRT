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

def RK4Simulate(state0,control):
	totalTime = 0.5
	h = 0.000025
	numberOfSteps = totalTime/h + 1

	time = np.linspace(0,totalTime,numberOfSteps).ravel()
	state = np.array(4)
	state = state0

	for i in np.arange(0,time.shape[0]-1):
		j = int(control_index(time[i]))
		k_1 = eom(time[i],state,control[j,:])
		k_2 = eom(time[i]+0.5*h,state+0.5*h*k_1,control[j,:])
		k_3 = eom((time[i]+0.5*h),(state+0.5*h*k_2),control[j,:])
		k_4 = eom((time[i]+h),(state+k_3*h),control[j,:])
		state = state + (1/6.0)*(k_1+2*k_2+2*k_3+k_4)*h  # main equation
	return state

def control_index(t):
	return int(t/0.025)

# def control_index(t):
# 	#dirty implementation - MORE ACCURATE!!!!
# 	if (0.0 <= t) and (t < 0.025):
# 		j = 1
# 	elif (0.025 <= t) and (t < 0.05):
# 		j = 2
# 	elif (0.05 <= t) and (t < 0.075):
# 		j = 3
# 	elif (0.075 <= t) and (t < 0.1):
# 		j = 4
# 	elif (0.1 <= t) and (t < 0.125):
# 		j = 5
# 	elif (0.125 <= t) and (t < 0.15):
# 		j = 6
# 	elif (0.15 <= t) and (t < 0.175):
# 		j = 7
# 	elif (0.175 <= t) and (t < 0.2):
# 		j = 8
# 	elif (0.2 <= t) and (t < 0.225):
# 		j = 9
# 	elif (0.225 <= t) and (t < 0.25):
# 		j = 10
# 	elif (0.25 <= t) and (t < 0.275):
# 		j = 11
# 	elif (0.275 <= t) and (t < 0.3):
# 		j = 12
# 	elif (0.3 <= t) and (t < 0.325):
# 		j = 13
# 	elif (0.325 <= t) and (t < 0.35):
# 		j = 14
# 	elif (0.35 <= t) and (t < 0.375):
# 		j = 15
# 	elif (0.375 <= t) and (t < 0.4):
# 		j = 16
# 	elif (0.4 <= t) and (t < 0.425):
# 		j = 17
# 	elif (0.425 <= t) and (t < 0.45):
# 		j = 18
# 	elif (0.45 <= t) and (t < 0.475):
# 		j = 19
# 	elif (0.475 <= t) and (t <= 0.5):
# 		j = 20
#
# 	return j-1
