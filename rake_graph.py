
from inferGraph import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

#import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.animation as an
import time

#Problem params
size = 10
timesteps = 100#8
samplesPerStep = 1
timeShift = 999 #Number of steps till new covariance matrix appears
eps = 1e-2
#Optimization parameters
alpha = 0.1 # Lasso parameter
beta = 3 # Weight between basis nodes



#Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)
np.random.seed(1)
ind_zero = spy.sparse.rand(size,size,0.25).todense()
S_true = (ind_zero + ind_zero.T)/2 + ind_zero.max()*np.matrix(np.eye(size))
Cov = alg.inv(S_true) # Covariance matrix #1

ind_zero2 = spy.sparse.rand(size,size,0.25).todense()
S_true2 = (ind_zero2 + ind_zero2.T)/2 + ind_zero2.max()*np.matrix(np.eye(size))
Cov2 = alg.inv(S_true2) # second covariance matrix



gvx = TGraphVX()

for i in range(timesteps):
	#Generate random samples
	x_samples = 0
	if (i < timeShift):
		x_samples = np.random.multivariate_normal(np.zeros(size), Cov, samplesPerStep).T
	else:
		x_samples = np.random.multivariate_normal(np.zeros(size), Cov2, samplesPerStep).T
	empCov = np.cov(x_samples)
 	if (samplesPerStep == 1):
	 	empCov = x_samples*x_samples.T

#	print empCov
	#Add Node, edge to previous timestamp
	n_id = i
	S = semidefinite(size,name='S')
	obj = -log_det(S) + trace(empCov*S) #+ alpha*norm(S,1)
	gvx.AddNode(n_id, obj)

	if (i > 0): #Add edge to previous timestamp
		prev_Nid = n_id - 1
		currVar = gvx.GetNodeVariables(n_id)
		prevVar = gvx.GetNodeVariables(prev_Nid)
		edge_obj = beta*norm(currVar['S'] - prevVar['S'],1)
		gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

	#Add rake nodes, edges
	gvx.AddNode(n_id + timesteps)
	gvx.AddEdge(n_id, n_id+timesteps, Objective=alpha*norm(S,1))


t = time.time()
gvx.Solve( NumProcessors = 1, MaxIters = 3)
end = time.time() - t
# t = time.time()
#gvx.Solve(UseADMM=False)
# end2 = time.time() - t
# print "TIMES", end, end2

#Plot results
# fig = pl.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# ims = []

#Print the matrix!
totalError = 0
for nodeID in range(timesteps):
	val = gvx.GetNodeValue(nodeID,'S')
	S_est = np.zeros((size,size))
	S_est[np.triu_indices(size)] = val #S_est matrix convert semidefinite (n(n-1) entries) to a full semidefinite matrix
	temp = S_est.diagonal()
	ind = (S_est<eps)&(S_est>-eps)
	S_est[ind]=0
	S_est = (S_est + S_est.T) - np.diag(temp)

	#Get actual InvCov
	S_actual = 0
	if (nodeID < timeShift):
		S_actual = S_true.copy()
	else:
		S_actual = S_true2.copy()

	# im1 = ax1.imshow(S_actual.copy(), interpolation='nearest', cmap = 'binary')
	# ax1.set_title('S_true')
	# im2 = ax2.imshow(S_est.copy(), interpolation='nearest', cmap = 'binary')
	# ax2.set_title('S(SnapVX)')
	# ims.append([im1, im2])

	# print "Timestamp", nodeID, ": Actual Inverse Covariance"
	# print S_actual
	# print "Predicted Inverse Covariance:"
	# print S_est
	totalError = totalError + np.linalg.norm(S_actual - np.matrix(S_est), 'fro')


print "Total Error: ", totalError
print "Total time: ", end
# ani = an.ArtistAnimation(fig, ims, interval=1000, repeat_delay = 3000)
# #ani.save('cov_animation.mp4')
# pl.show()
   






