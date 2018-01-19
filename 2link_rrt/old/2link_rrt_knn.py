import rrt_knn
import time
import numpy as numpy

NUM_PLANNING_ATTEMPTS = 1
START_STATE	= numpy.array([0.,0.,0.,0.])
GOAL_STATE	= numpy.array([numpy.pi,0.,0.,0.])
# GOAL_STATE	= numpy.array([1.,1.,1.,1.])
NUM_NODES	= 100
STATE_DIMENSION = 4
STATE_RANGE = numpy.array([[0,0,-30,-30],[2*numpy.pi,2*numpy.pi,30,30]])
GOAL_TOLERANCE 	= 1

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
		newTreeNode = rrt_knn.TreeNode()
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
			newTreeNode = rrt_knn.TreeNode()
			# Find the nearest neighbour to connect to
			foundValidPrediction = False
			while foundValidPrediction == False:
				# Sample a new random state from state space.
				rState = rrt_knn.sampleState(STATE_DIMENSION, STATE_RANGE,GOAL_STATE,GOAL_TOLERANCE)
				# print rState
				# Find the nearest neighbor
				newTreeNode.parentNode, newTreeNode.costToGo, foundValidPrediction = \
					rrt_knn.findNearestNeighbor(nodeList,rState)

			# If the new state is within tolerance of the goal
			if numpy.linalg.norm(rState - pGoal) < GOAL_TOLERANCE :
				# TODO: what is happening here?
				randomFactorGoal = randomFactorGoal + 0.5
				randomFactorCurrent = randomFactorGoal
			else:
				randomFactorCurrent = randomFactorAny

			# Connect the neighbor to the node
			newTreeNode.childNode, newTreeNode.coState, newTreeNode.costToGo, connectionValid = \
				rrt_knn.connectNodes(newTreeNode.parentNode, rState,randomFactorCurrent)
			if connectionValid:
				print idx
				# Add the node to the tree
				treeNodes.append(newTreeNode)
				# Add the new node to the list of available nodes
				nodeList = numpy.vstack((nodeList, newTreeNode.childNode))

				goalReach = rrt_knn.goalReached(newTreeNode.childNode, pGoal)
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
