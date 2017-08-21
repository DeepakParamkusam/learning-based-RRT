import numpy
import plant
from sklearn.externals import joblib

kNN_control = joblib.load('../2link_NN/trained_models/knn_control_ft_2_10k_test')
kNN_cost = joblib.load('../2link_NN/trained_models/knn_cost_2_10k')
control_training = numpy.loadtxt('../2link_NN/training_data/2_control_data_fulltraj_10k.txt')
cost_training = numpy.loadtxt('../2link_NN/training_data/2_cost_10k.txt')

NUMBER_OF_NEIGHBOURS = 3
NEIGHBOUR_BOUND = 100
STATE_DIMENSION = 4

class TreeNode(object):
	parentNode 	= None
	childNode	= None
	costToGo	= None
	coState		= None

def neighPredict(Xdata):
	distances, neighbours = numpy.array(kNN_cost.kneighbors(Xdata,5,return_distance=True))
	neighbours = neighbours.astype(int)
	cost_neigh = cost_training[neighbours,-1]
	cost_pred=numpy.average(cost_neigh,1)

	return numpy.vstack((distances[:,-1],cost_pred))

def modelPredict(initialState,goalState):
	inp = numpy.hstack((initialState,goalState))
	input2NN = numpy.divide(inp - kNN_control[1][0],kNN_control[1][1])

	predicted_cost = kNN_cost.predict(input2NN)
	predicted_control = kNN_control[0].predict(input2NN)

	predicted_cost = predicted_cost[0]*kNN_control[1][1] + kNN_control[1][0]
	predicted_control = predicted_control[0]*kNN_control[1][3] + kNN_control[1][2]
	predicted_control = numpy.transpose(numpy.vstack([predicted_control[0:20],predicted_control[20:40]]))

	return (predicted_cost,predicted_control)

def connectNodes(initialState, goalState):
	connectionValid = False
	# Predict cost to go and input.
	cost_ann,costates_ann  = modelPredict(initialState, goalState)

	finalCost = cost_ann
	finalCostate = costates_ann
	print finalCostate
	# print finalCostate
	finalState = plant.RK4Simulate(initialState, finalCostate)

	# Connection validity already established earlier.
	stateError = numpy.linalg.norm(finalState - goalState)

	a = finalState / numpy.linalg.norm(finalState)
	b = goalState  / numpy.linalg.norm(goalState)
	angle = numpy.arccos(numpy.clip(numpy.dot(a.ravel(),b.ravel()),-1.,1.))
	if stateError < 1:
		print "angle: ",numpy.degrees(angle),"\terror: ",stateError
	if stateError < 0.3 and numpy.degrees(angle) < 0.5:
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
	# print allPredictions

	# Locate all nodes that would lead to valid predictions.
	validNodes = allPredictions[:,0] < NEIGHBOUR_BOUND
	print validNodes

	# Nodes that lead to valid predictions exist, continue further.
	if validNodes.any():
		# Compute the predictions based on Neural nets.
		#cost,coState = rrt_predict(allData, model)
		if len(nl_size) == 1:
			cost = allPredictions[1]
		else:
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
	if distanceToGoal < 0.2:
		return True
	else:
		return False
