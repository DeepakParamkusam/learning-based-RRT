import rrt
import time
import numpy as np

NUM_PLANNING_ATTEMPTS = 1
START_STATE	= np.array([0.,0.,0.,0.])
GOAL_STATE	= np.array([np.pi,0.,0.,0.])
NUM_NODES	= 10000
STATE_DIMENSION = 4
STATE_RANGE = np.array([[-3/2*np.pi,-3/2*np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi,np.pi]])
COSTATE_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01,np.pi*3/2-0.01]])
PHI_RANGE = np.array([[-np.pi/2+0.01,-np.pi/2+0.01,-np.pi/2+0.01],[3*np.pi/2-0.01,3*np.pi/2-0.01,3*np.pi/2-0.01]])
GOAL_TOLERANCE 	= 0.15

if __name__ == "__main__":
	planning_timeout = 0
	planning_times = []
	planning_nodes = []
	planning_times_succesful = []
	planning_nodes_succesful = []

	print "Running RRT-CoLearn..."

	# Attempt $NUM_PLANNING_ATTEMPTS times
	for numPlanningAttempts in range(NUM_PLANNING_ATTEMPTS):
		# Update the progress bar
		# Create an empty tree
		goalReach = False
		treeNodes = []
		# Create an empty list of all the nodes. Maintain a phase space tree
		# because the training data is in phase space.

		pInit = START_STATE
		nodeList = pInit
		# nodeList = np.reshape(nodeList,[1,4])

		# Convert goal state from state space to phase space.
		pGoal = GOAL_STATE

		# Construct the starting node
		newTreeNode = rrt.TreeNode()
		newTreeNode.parentNode = pInit
		newTreeNode.childNode  = pInit
		newTreeNode.costToGo = 0.0
		newTreeNode.coState = 0.0
		# Add the starting node of the tree
		treeNodes.append(newTreeNode)

		# Set randomization parameter
		# TODO: What is this?
		randomFactorAny = 0.2
		randomFactorGoal = 0.2

		# Build the RRT until goal is reached or max number of nodes are reached
		time1 = time.time()
		idx = 2

		for idx in range(NUM_NODES):
			newTreeNode = rrt.TreeNode()
			# Find the nearest neighbour to connect to
			foundValidPrediction = False
			while foundValidPrediction == False:
				# Sample a new random state from state space.
				rState = rrt.sampleState(STATE_DIMENSION, STATE_RANGE,GOAL_STATE,GOAL_TOLERANCE)

				# Find the nearest neighbor
				newTreeNode.parentNode, newTreeNode.costToGo, foundValidPrediction = \
					rrt.findNearestNeighbor(nodeList,rState)
			# If the new state is within tolerance of the goal
			if np.linalg.norm(rState - pGoal) < GOAL_TOLERANCE :
				# TODO: what is happening here?
				randomFactorGoal = randomFactorGoal + 0.5
				randomFactorCurrent = randomFactorGoal
			else:
				randomFactorCurrent = randomFactorAny

			# Connect the neighbor to the node
			newTreeNode.childNode, newTreeNode.coState, newTreeNode.costToGo, connectionValid = \
				rrt.connectNodes(newTreeNode.parentNode, rState)
			if connectionValid:
				# Add the node to the tree
				treeNodes.append(newTreeNode)
				# Add the new node to the list of available nodes
				nodeList = np.vstack((nodeList, newTreeNode.childNode))

				goalReach = rrt.goalReached(newTreeNode.childNode, pGoal)
				if goalReach:
					print "final node: ",newTreeNode.childNode
					time2 = time.time()
					planning_time = (time2 - time1)
					planning_times.append(planning_time)
					planning_nodes.append(nodeList.shape[1])

					print "Planning successful!"
					print "Planning time is %0.3f s" %(planning_time)
					print "Goal reached. No further tree nodes will be added."
					print "Number of tree nodes: ", (nodeList.shape[0])

					# Print the path
					completePath = treeNodes[-1].childNode
					parentIndex = np.where(nodeList == treeNodes[-1].parentNode)[0]
					while parentIndex[0] != 0:
						completePath = np.vstack((completePath, treeNodes[parentIndex[0]].childNode))
						parentIndex = np.where(nodeList == treeNodes[parentIndex[0]].parentNode)[0]
						print parentIndex[0]
					completePath = np.vstack((completePath, treeNodes[parentIndex[0]].childNode))
					print completePath
					pathLength = completePath.shape[0]
					print "Path length = ",pathLength
					np.savetxt('completePath.txt',completePath,delimiter=',')

					while pathLength >= 2:
						srcDst1 = np.array([completePath[pathLength-1][0],completePath[pathLength-2][0]])
						print pathLength
						srcDst2 = np.array([completePath[pathLength-1][1],completePath[pathLength-2][1]])
						pathLength = pathLength - 1

					break
