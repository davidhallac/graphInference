from inferGraph import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

#import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#import matplotlib.animation as an
import time
# for size 10, alph = 0.1 beta = 0.6
# Problem parameters
size = 5
timesteps = 5# size/2
useCVX = False

samplesPerStep = 10#int(np.log2(size))
timeShift = timesteps/3 #Number of steps till new covariance matrix appears
eps = 1e-3

# Optimization parameters
alpha = 0.1 # Lasso parameter
beta = 0.6 # Weight between basis nodes
#set_length = 1
#alpha_set = np.linspace(0.1,10,set_length)
#beta_set = np.linspace(3,10,set_length)
FroError = []
Score = []

# Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)

np.set_printoptions(suppress=True, precision = 3, threshold = 5)
S_true = np.zeros((size,size))
while (alg.det(S_true) <= 1e-2 ):
    #print int(numpy.log2(size))*size
    G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
    #G6 = snap.GenRndGnm(snap.PUNGraph, 5, 5)
    S_true = numpy.zeros((size,size))
    for EI in G6.Edges():
        S_true[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 #  np.random.rand(1) #2*(numpy.random.rand(1)-0.5)
    #    S[EI.GetSrcNId(), EI.GetDstNId()] =  2*(np.random.rand(1)-0.5)
#        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    #print S, S.max(axis = 1)
    #S =  S + S.T + np.diag( S.max(axis = 1) )#S.max()*numpy.matrix(numpy.eye(size))
    S_true =  S_true + S_true.T + S_true.max()*np.matrix(np.eye(size))
#    print S_true
#    print alg.det(S_true)
Cov = alg.inv(S_true)
S_true2 = numpy.zeros((size,size))
while (alg.det(S_true2) <= 1e-2 ):
    #print int(numpy.log2(size))*size
    G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
    #G6 = snap.GenRndGnm(snap.PUNGraph, 5, 5)
    S_true2 = np.zeros((size,size))
    for EI in G6.Edges():
        S_true2[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 #  np.random.rand(1) #2*(numpy.random.rand(1)-0.5)
    #    S[EI.GetSrcNId(), EI.GetDstNId()] =  2*(np.random.rand(1)-0.5)
#        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    #print S, S.max(axis = 1)
    #S =  S + S.T + np.diag( S.max(axis = 1) )#S.max()*numpy.matrix(numpy.eye(size))
    S_true2 =  S_true2 + S_true2.T + S_true2.max()*np.matrix(np.eye(size))
#    print S
#    print alg.det(S_true2)
Cov2 = alg.inv(S_true2)
S_true3 = numpy.zeros((size,size))
while (alg.det(S_true3) <= 1e-2 ):
    #print int(numpy.log2(size))*size
    G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
    #G6 = snap.GenRndGnm(snap.PUNGraph, 5, 5)
    S_true3 = np.zeros((size,size))
    for EI in G6.Edges():
        S_true3[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 #  np.random.rand(1) #2*(numpy.random.rand(1)-0.5)
    #    S[EI.GetSrcNId(), EI.GetDstNId()] =  2*(np.random.rand(1)-0.5)
#        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    #print S, S.max(axis = 1)
    #S =  S + S.T + np.diag( S.max(axis = 1) )#S.max()*numpy.matrix(numpy.eye(size))
    S_true3 =  S_true3 + S_true3.T + S_true3.max()*np.matrix(np.eye(size))
#    print S
#    print alg.det(S_true2)
Cov3 = alg.inv(S_true3)
#np.random.seed(1)
#ind_zero = spy.sparse.rand(size,size,0.25).todense()
#S_true = (ind_zero + ind_zero.T)/2 + ind_zero.max()*np.matrix(np.eye(size))
#Cov = alg.inv(S_true) # Covariance matrix #1
#
#ind_zero2 = spy.sparse.rand(size,size,0.25).todense()
#S_true2 = (ind_zero2 + ind_zero2.T)/2 + ind_zero2.max()*np.matrix(np.eye(size))
#Cov2 = alg.inv(S_true2) # second covariance matrix

#
#for alpha in alpha_set:
#    for beta in beta_set:
#        print alpha, beta
gvx = TGraphVX()
for i in range(timesteps):
	# Generate random samples
	x_samples = 0
	if (i < timeShift):
		x_samples = np.random.multivariate_normal(np.zeros(size), Cov, samplesPerStep).T
	elif (i < 2*timeShift):
		x_samples = np.random.multivariate_normal(np.zeros(size), Cov2, samplesPerStep).T
	else:
		x_samples = np.random.multivariate_normal(np.zeros(size), Cov3, samplesPerStep).T
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
		edge_obj = beta*norm(currVar['S'] - prevVar['S'],1) # one norm penalty function
#		edge_obj = beta*norm(currVar['S'] - prevVar['S'],2) # two norm penalty function
		gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

	#Add rake nodes, edges
	gvx.AddNode(n_id + timesteps)
	gvx.AddEdge(n_id, n_id+timesteps, Objective=alpha*norm(S,1))

print "Starting to solve:"
t = time.time()
if(useCVX):
    gvx.Solve( UseADMM = False)
else:
    gvx.Solve(Rho = 0.5, Verbose=True)

end = time.time() - t
print "SOLUTION TIME", end





