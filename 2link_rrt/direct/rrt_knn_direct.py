import numpy
import plant
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

#CONSTRAINED MODEL
kNN_control = joblib.load('../../trained_models/knn-direct-cons-control.pkl')
kNN_cost = joblib.load('../../trained_models/knn-direct-cons-cost.pkl')
cost_training = numpy.loadtxt('../../training_data/clean_direct/cost_lagrange_cons_clean.txt',delimiter=',')

#UNCONSTRAINED MODEL
# kNN_control = joblib.load('../../trained_models/knn-direct-uncons-control.pkl')
# kNN_cost = joblib.load('../../trained_models/knn-direct-uncons-cost.pkl')
# cost_training = numpy.loadtxt('../../training_data/clean_direct/cost_jan12_bound.txt',delimiter=',')

xscaler = StandardScaler()
xscaler.fit(cost_training[:,0:8])

NUMBER_OF_NEIGHBOURS = 3
NEIGHBOUR_BOUND = 5
STATE_DIMENSION = 4
NUM_PLANNING_ATTEMPTS = 1
START_STATE	= numpy.array([0.,0.,0.,0.])
GOAL_STATE	= numpy.array([numpy.pi,0.,0.,0.])
# GOAL_STATE	= numpy.array([1.,1.,1.,1.])
NUM_NODES	= 100
STATE_DIMENSION = 4
STATE_RANGE = numpy.array([[0,0,-30,-30],[2*numpy.pi,2*numpy.pi,30,30]])
GOAL_TOLERANCE 	= 1

class TreeNode(object):
	parentNode 	= None
	childNode	= None
	costToGo	= None
	coState		= None

def neighPredict(Xdata):
	distances, neighbours = numpy.array(kNN_cost.kneighbors(Xdata,5,return_distance=True))
	neighbours = neighbours.astype(int)
	
	class_pred = distances[:,-1]
	cost_neigh = cost_training[neighbours,-1]
	
	cost_pred = numpy.average(cost_neigh,1)
	# print numpy.hstack((class_pred[:,None],cost_pred[:,None]))
	return numpy.hstack((class_pred[:,None],cost_pred[:,None]))

def modelPredict(initialState,goalState):
	inp = numpy.hstack((initialState,goalState))
	input2NN = xscaler.transform(inp)
	input2NN = input2NN.reshape(1,-1)

	predicted_cost = kNN_cost.predict(input2NN)
	predicted_control = kNN_control.predict(input2NN)

	predicted_cost = predicted_cost[0]
	predicted_control = predicted_control[0]
	
	# print predicted_cost
	predicted_control = numpy.transpose(numpy.vstack([predicted_control[0:20],predicted_control[20:40]]))

	return (predicted_cost,predicted_control)

def connectNodes(initialState, goalState,randomFactor):
	connectionValid = False
	# Predict cost to go and input.
	cost_ann,costates_ann  = modelPredict(initialState, goalState)
	
	# finalCost = cost_ann
	# finalCostate = costates_ann
	finalCost = (cost_ann + randomFactor*numpy.random.uniform(0,1))/(1+randomFactor)
	finalCostate = (costates_ann+randomFactor*numpy.random.uniform(0,2*numpy.pi)) / (1+randomFactor)
	finalState = plant.RK4Simulate(initialState, finalCostate)

	# Connection validity already established earlier.
	stateError = numpy.linalg.norm(finalState - goalState)
	# print stateError

	a = finalState/numpy.linalg.norm(finalState)
	b = goalState/numpy.linalg.norm(goalState)
	angle = numpy.arccos(numpy.clip(numpy.dot(a.ravel(),b.ravel()),-1.,1.))
	if stateError < 2:
		print "angle: ",numpy.degrees(angle),"\terror: ",stateError
	if stateError < 2 and numpy.degrees(angle) < 30:
		connectionValid = True
	else:
		connectionValid = False

	return (finalState, finalCostate, finalCost, connectionValid)

def sampleState(stateDim,stateRange,goalState,goalTolerance):
	# Take a random sample 80% of the time
	if numpy.random.uniform(0, 1) < 0.8:
		rState  	= numpy.random.uniform(stateRange[0],stateRange[1])
	# Take a sample near the goal 20% of the time
	else:
		rState 	= goalState + numpy.random.uniform(-goalTolerance,goalTolerance,stateDim)
	# Return the full random state
	return rState
	# return numpy.ones(4)

# Sort only valid values from an array containing invalid values,
# return the original location of indices without altering original array size.
def sparse_argsort(arr, validNodes):
	indices = numpy.where(validNodes)[0].flatten()
	print indices
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
	validNodes = allPredictions[:,0] < NEIGHBOUR_BOUND
	
	# Nodes that lead to valid predictions exist, continue further.
	if validNodes.any():
		# Compute the predictions based on Neural nets.
		#cost,coState = rrt_predict(allData, model)
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
	if distanceToGoal < 1:
		return True
	else:
		return False

if __name__ == "__main__":
	print "Running RRT-CoLearn..."

	# Attempt $NUM_PLANNING_ATTEMPTS times
	for numPlanningAttempts in range(NUM_PLANNING_ATTEMPTS):
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
		idx = 2

		for idx in range(NUM_NODES):
			# print idx
			newTreeNode = TreeNode()
			# Find the nearest neighbour to connect to
			foundValidPrediction = False
			while foundValidPrediction == False:
				# Sample a new random state from state space.
				rState = sampleState(STATE_DIMENSION, STATE_RANGE,GOAL_STATE,GOAL_TOLERANCE)
				# print rState
				# Find the nearest neighbor
				newTreeNode.parentNode, newTreeNode.costToGo, foundValidPrediction = \
					findNearestNeighbor(nodeList,rState)

			# If the new state is within tolerance of the goal
			if numpy.linalg.norm(rState - pGoal) < GOAL_TOLERANCE :
				# TODO: what is happening here?
				randomFactorGoal = randomFactorGoal + 0.5
				randomFactorCurrent = randomFactorGoal
			else:
				randomFactorCurrent = randomFactorAny

			# Connect the neighbor to the node
			newTreeNode.childNode, newTreeNode.coState, newTreeNode.costToGo, connectionValid = \
				connectNodes(newTreeNode.parentNode, rState,randomFactorCurrent)
			if connectionValid:
				print idx
				# Add the node to the tree
				treeNodes.append(newTreeNode)
				# Add the new node to the list of available nodes
				nodeList = numpy.vstack((nodeList, newTreeNode.childNode))

				goalReach = goalReached(newTreeNode.childNode, pGoal)
				if goalReach:
					print "final node: ",newTreeNode.childNode
					time2 = time.time()
					planning_time = (time2 - time1)
					print planning_time
					# planning_times.append(planning_time)
					# planning_nodes.append(nodeList.shape[1])
					#
					# print "Planning successful!"
					# print "Planning time is %0.3f s" %(planning_time)
					# print "Goal reached. No further tree nodes will be added."
					# print "Number of tree nodes: ", (nodeList.shape[0])
					#
					# Print the path
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
					numpy.savetxt('completePath.txt',completePath,delimiter=',')
					#
					# while pathLength >= 2:
					# 	srcDst1 = numpy.array([completePath[pathLength-1][0],completePath[pathLength-2][0]])
					# 	print pathLength
					# 	srcDst2 = numpy.array([completePath[pathLength-1][1],completePath[pathLength-2][1]])
					# 	pathLength = pathLength - 1
					#
					break
