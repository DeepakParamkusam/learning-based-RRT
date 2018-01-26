import time
import numpy
import random
import plant_indirect
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

mlp_cost = joblib.load('mlp-indirect-10b-uncons-cost-100-100.pkl')
kNN_cost = joblib.load('knn-indirect-10lb-uncons-cost.pkl')
x = numpy.loadtxt('X10b.txt',delimiter=',')

xscaler = StandardScaler()
xscaler.fit(x)

NUM_NODES	= 1000
STATE_DIMENSION = 4
NEIGHBOUR_BOUND = 9
STATE_DIMENSION = 4
NUMBER_OF_NEIGHBOURS = 3
STATE_RANGE = numpy.array([[0,0,-30,-30],[2*numpy.pi,2*numpy.pi,30,30]])
PHI_RANGE = numpy.array([[-numpy.pi/2+0.01,-numpy.pi/2+0.01,-numpy.pi/2+0.01],[3*numpy.pi/2-0.01,3*numpy.pi/2-0.01,3*numpy.pi/2-0.01]])
COSTATE_RANGE = numpy.array([[-1.5607613 , -1.56063197, -1.5607664],[4.70237843e+00, 4.70235522e+00, 4.70233914e+00]])
FSTATE_RANGE = numpy.array([[-2.15450000e+00,  -1.77740000e+00,-3.12380000e+01,  -3.16480000e+01],[8.1928,   8.2597, 31.616 ,  31.703]])

NUM_PLANNING_ATTEMPTS = 1
START_STATE	= numpy.array([0.,0.,0.,0.])
# GOAL_STATE	= numpy.array([numpy.pi,0.,0.,0.])
# GOAL_STATE	= numpy.array([numpy.pi,numpy.pi,5.,5.])
GOAL_STATE	= numpy.array([numpy.pi,0,10.,-5.])
GOAL_TOLERANCE 	= 0.5

class TreeNode(object):
	parentNode 	= None
	childNode	= None
	costToGo	= None
	coState		= None

def neighPredict(Xdata):
	distances, neighbours = numpy.array(kNN_cost.kneighbors(xscaler.transform(Xdata),4,return_distance=True))
	class_pred = distances[:,-1]
	cost_pred = numpy.array(mlp_cost.predict(Xdata))
	return numpy.hstack((class_pred[:,None],cost_pred[:,None]))

def modelPredict(initialState,goalState):
	inp = numpy.hstack((initialState,goalState))
	input2NN = xscaler.transform(inp)
	input2NN = input2NN.reshape(1,-1)

	predicted_cost = mlp_cost.predict(input2NN)
	return predicted_cost[0]

def connectNodes(initialState, goalState,randomFactor):
	finalCost = modelPredict(initialState, goalState)
	# finalCost = (modelPredict(initialState, goalState)+ randomFactor*numpy.random.uniform(0,1))/(1+randomFactor)
	print initialState
	print goalState
	print finalCost	
	# Generate 3 random costates, simulate system with them to new state and choose the one closest to goal
	finalDist2goal = 10000
	for rand_cont_idx in range(3):
		# rand_finalCostate = genRandomCostate(initialState,PHI_RANGE)
		rand_finalCostate = (genRandomCostate(initialState)+randomFactor*numpy.random.uniform(0,2*numpy.pi)) / (1+randomFactor)		
		rand_finalState = plant_indirect.plantRRTSimulate(finalCost,initialState,rand_finalCostate)
		rand_finalState[0] = rand_finalState[0]%(2*numpy.pi)  
		rand_finalState[1] = rand_finalState[1]%(2*numpy.pi)
		rand_dist2goal = numpy.linalg.norm(rand_finalState - goalState)

		if inFStateRange(rand_finalState):
			connectionValid = True
			if rand_cont_idx == 0:
				finalState = rand_finalState
				finalCostate = rand_finalCostate
				finalDist2goal = rand_dist2goal
			else:
				if rand_dist2goal < finalDist2goal:
					finalState = rand_finalState
					finalCostate = rand_finalCostate
		else:
			connectionValid = False
			finalState = rand_finalState
			finalCostate = rand_finalCostate

	# print finalState
	# print connectionValid
	# print connectionValid
	return (finalState, finalCostate, finalCost, connectionValid)

def sampleState(stateDim,stateRange,goalState,goalTolerance):
	# Take a random sample 70% of the time
	if numpy.random.uniform(0, 1) < 0.7:
		rState  	= numpy.random.uniform(stateRange[0],stateRange[1])
	# Take a sample near the goal 30% of the time
	else:
		# rState 	= goalState + numpy.random.uniform(-goalTolerance,goalTolerance,stateDim)
		rState 	= goalState

	# Return the full random state
	return rState
	
# Sort only valid values from an array containing invalid values,
# return the original location of indices without altering original array size.
def sparse_argsort(arr, validNodes):
	indices = numpy.where(validNodes)[0].flatten()
	return indices[numpy.argsort(arr[indices])]

def findNearestNeighbor(nodeList, rState):
	nl_size = nodeList.shape
	if len(nl_size) == 1:
		t_nodeList = numpy.zeros([1,4])
		t_nodeList[0] = nodeList
	else:
		t_nodeList = nodeList

	rStateMatrix = numpy.matlib.repmat(rState,t_nodeList.shape[0],1)
	allData = numpy.hstack((t_nodeList, rStateMatrix))
	
	# Filter out nodes that would lead to invalid prediction based on nearest neighbors.
	# This means, the randomly sampled node is too far from the trained model.
	allPredictions = neighPredict(allData)
	
	# Locate all nodes that would lead to valid predictions.
	non_zero_cost = allPredictions[:,1] > 0
	lt_neighbound = allPredictions[:,0] < NEIGHBOUR_BOUND
	validNodes = numpy.logical_and(non_zero_cost,lt_neighbound)
	
	# Nodes that lead to valid predictions exist, continue further.
	if validNodes.any():
		# Compute the predictions based on Neural nets.
		cost = allPredictions[:,1]

		# Get the indices of potential nearest neighbors.
		nnIdces = sparse_argsort(cost, validNodes)[:NUMBER_OF_NEIGHBOURS]

		# Pick a random neighbor from the potential neighbors.
		nnIdx = numpy.random.randint(0,len(nnIdces))
		neighbor = t_nodeList[nnIdces[nnIdx],:]

		# Get the cost-to-go from neighbor to the random state.
		costChosen = cost[nnIdces[nnIdx]]
		foundValidPrediction = True
	else:
		foundValidPrediction = False
		neighbor = numpy.zeros((1,STATE_DIMENSION))
		costChosen = 0.0
	return neighbor,costChosen,foundValidPrediction

def goalReached(currentState, goalState):
	distanceToGoal = numpy.linalg.norm(currentState-goalState)
	print "Distance to goal from last added node=",distanceToGoal

	if distanceToGoal < 1:
		return True
	else:
		return False

def genRandomCostate(x0):
	a = random.choice([0,1])
	phi0 = numpy.random.uniform(COSTATE_RANGE[0], COSTATE_RANGE[1])
	return numpy.append(a,phi0)

def inFStateRange(final_state):
	# print final_state
	lb = final_state > FSTATE_RANGE[0]
	ub = final_state < FSTATE_RANGE[1]
	valid = numpy.logical_and(lb,ub)
	if numpy.all(valid) == True:
		# print "False"
		return True
	else:
		return False

if __name__ == "__main__":
	print "Running RRT-CoLearn..."
	SUCCESS_ATTEMPTS = 0

	# Attempt $NUM_PLANNING_ATTEMPTS times
	for numPlanningAttempts in range(NUM_PLANNING_ATTEMPTS):
		print "Attempt=",numPlanningAttempts
		goalReach = False
		treeNodes = []
		# Create an empty list of all the nodes.

		pInit = START_STATE
		nodeList = pInit
		pGoal = GOAL_STATE

		# Construct the starting node
		newTreeNode = TreeNode()
		newTreeNode.parentNode = pInit
		newTreeNode.childNode  = pInit
		newTreeNode.costToGo = 0.0
		newTreeNode.coState = 0.0
		# Add the starting node of the tree
		treeNodes.append(newTreeNode)

		# Set randomization parameter
		randomFactorAny = 0.2
		randomFactorGoal = 0.2

		# Build the RRT until goal is reached or max number of nodes are reached
		time1 = time.time()
		# print time1
		idx = 2

		for idx in range(NUM_NODES):
			newTreeNode = TreeNode()
			# Find the nearest neighbour to connect to
			foundValidPrediction = False
			while foundValidPrediction == False:
				# Sample a new random state from state space.
				rState = sampleState(STATE_DIMENSION, FSTATE_RANGE,GOAL_STATE,GOAL_TOLERANCE)
				# Find the nearest neighbor
				newTreeNode.parentNode, newTreeNode.costToGo, foundValidPrediction = \
					findNearestNeighbor(nodeList,rState)

			# If the new state is within tolerance of the goal
			if numpy.linalg.norm(rState - pGoal) < GOAL_TOLERANCE :
				randomFactorGoal = randomFactorGoal + 0.5
				randomFactorCurrent = randomFactorGoal
			else:
				randomFactorCurrent = randomFactorAny

			# Connect the neighbor to the node
			newTreeNode.childNode, newTreeNode.coState, newTreeNode.costToGo, connectionValid = \
				connectNodes(newTreeNode.parentNode, rState,randomFactorCurrent)
			if connectionValid:
				print "Nodes in tree=",len(treeNodes)
				# Add the node to the tree
				treeNodes.append(newTreeNode)
				# Add the new node to the list of available nodes
				nodeList = numpy.vstack((nodeList, newTreeNode.childNode))

				goalReach = goalReached(newTreeNode.childNode, pGoal)
				if goalReach:
					print "Planning successful"
					print "final node: ",newTreeNode.childNode
					time2 = time.time()
					planning_time = (time2 - time1)
					print planning_time
					
					completePath = treeNodes[-1].childNode
					parentIndex = numpy.where(nodeList == treeNodes[-1].parentNode)[0]
					while parentIndex[0] != 0:
						completePath = numpy.vstack((completePath, treeNodes[parentIndex[0]].childNode))
						parentIndex = numpy.where(nodeList == treeNodes[parentIndex[0]].parentNode)[0]
						print parentIndex[0]
					completePath = numpy.vstack((completePath, treeNodes[parentIndex[0]].childNode))
					print completePath
					pathLength = completePath.shape[0]
					print "Path length = ",pathLength
					numpy.savetxt('solution/completePath.txt',completePath,delimiter=',')
					SUCCESS_ATTEMPTS = SUCCESS_ATTEMPTS + 1
					
					break
		joblib.dump(treeNodes,'solution/treeNodes.pkl')
		print "time_per attempt=",(time.time()-time1)
	print SUCCESS_ATTEMPTS	