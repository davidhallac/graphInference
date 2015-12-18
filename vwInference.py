
from inferGraph import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

#import matplotlib.pyplot as plt
#import matplotlib.pylab as pl
# #import matplotlib.animation as an
import time
from itertools import *
import sys

samplesPerStep = 10
numFeatures = 9 
beta = 10
lamb = 100
eps = 1e-2 #Treat these as essentially 0


#Read in file
filename = "/dfs/scratch0/abhisg/granular_analysis/new_data/0/2014-03-10/snapshots_0_2014-03-10_0_4.dat"
if (sys.platform == 'darwin'):
	filename = "/Users/Hallac/Desktop/vw_data.dat"

for line in open(filename, 'r'):
	item = line.rstrip()
gvx = TGraphVX()

nodeNum = 0
with open(filename,'r') as f:
	while True:
		sample = list(islice(f, samplesPerStep))
		if not sample:
			break
		#Now we have n samples of time
		#[u'Day + Time', u'Steer_Angle', u'Velocity', u'Heading', u'Latitude', u'Longitude', u'Brightness', 
		# (Index 7) u'Road_Type', u'Num_Oncoming_Lanes', u'Curr_Seg_ID', u'Seg_Dist_Left', u'Next_Seg_ID', u'Steer_Velocity',
		# (Index 13) u'Brake', u'Pedal', u'Fuel', u'X_Accel', u'Y_Accel', u'Mileage', u'RPM', u'Wiper_Speed', u'Time2Coll', 
		#u'Deacc_Request', u'L_Sig', u'R_Sig', u'Distance', u'User', u'Session', u'State'],dtype='object')
		readings = np.zeros(shape=(samplesPerStep,numFeatures))
		counter = 0
		for line in sample:
			temp = line.rstrip().split('\t')
			#Steer_Angle, Velocity, Heading, Steer_Velocity, Brake, Pedal, X_Accel, Y_Accel, RPM
			#region = [temp[1], temp[2], temp[3], temp[12], temp[13],temp[14],temp[16],temp[17],temp[19]]
			region = [temp[2], temp[3], temp[4], temp[13], temp[14],temp[15],temp[17],temp[18],temp[20]]
			readings[counter,:] = region
			counter = counter+1

		#Get empirical covariance
		empCov = 0
		if (samplesPerStep > 1):
			empCov = np.cov(readings.T)
		else:
			empCov = readings.T*readings

		#Add node to graph
		S = semidefinite(numFeatures,name='S')
		gvx.AddNode(nodeNum, -log_det(S) + trace(empCov*S))

		if (nodeNum > 0): #Add edge to previous timestamp
			prev_Nid = nodeNum - 1
			currVar = gvx.GetNodeVariables(nodeNum)
			prevVar = gvx.GetNodeVariables(prev_Nid)
			edge_obj = beta*norm(currVar['S'] - prevVar['S'],1)
			gvx.AddEdge(nodeNum, prev_Nid, Objective=edge_obj)


		nodeNum = nodeNum + 1

totalNodes = gvx.GetNodes()
for i in range(totalNodes):
	nodeVar = gvx.GetNodeVariables(i)
	gvx.AddNode(i + totalNodes)
	gvx.AddEdge(i, i+totalNodes, Objective=lamb*norm(nodeVar['S'],1))

#Solve the problem
gvx.Solve()




#Print the solution
for nodeID in range(totalNodes-1):
	if (nodeID == 0):
		val = gvx.GetNodeValue(nodeID,'S')
		S_est = np.zeros((numFeatures,numFeatures))
		S_est[np.triu_indices(numFeatures)] = val
		temp = S_est.diagonal()
		ind = (S_est<eps)&(S_est>-eps)
		S_est[ind]=0
		S_est = (S_est + S_est.T) - np.diag(temp)
	else:
		S_est = S_next

	nextVal = gvx.GetNodeValue(nodeID+1,'S')
	S_next = np.zeros((numFeatures,numFeatures))
	S_next[np.triu_indices(numFeatures)] = nextVal
	temp = S_next.diagonal()
	ind = (S_next<eps)&(S_next>-eps)
	S_next[ind]=0
	S_next = (S_next + S_next.T) - np.diag(temp)

	diff = alg.norm(S_next - S_est, 'fro')
	if (diff < 1e-4):
		print "Node: ", nodeID, "; Difference = 0"
	else:
		print "Node: ", nodeID, "; Difference = ", alg.norm(S_next - S_est, 'fro')







