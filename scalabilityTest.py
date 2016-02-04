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
size = 50
timesteps = 50# size/2
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
#gvx.Solve(Rho = 0.5, Verbose=True)
gvx.Solve( UseADMM = False)
end = time.time() - t
print "SOLUTION TIME", end
# t = time.time()
#gvx.Solve(UseADMM=False)
# end2 = time.time() - t
# print "TIMES", end, end2

#Plot results
# fig = pl.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# ims = []


# e1 = 0
# e2 = 0
# e1_set = []
# e2_set = []
# for nodeID in range(timesteps):
#     val = gvx.GetNodeValue(nodeID,'S')
#     S_est = np.zeros((size,size))
#     S_est[np.triu_indices(size)] = val #S_est matrix convert semidefinite (n(n-1) entries) to a full semidefinite matrix
#     temp = S_est.diagonal()
#     ind = (S_est<eps)&(S_est>-eps)
#     S_est[ind]=0
#     S_est = np.asarray((S_est + S_est.T) - np.diag(temp))
        
#     #Get actual InvCov
#     S_actual = 0
#     if (nodeID < timeShift):
#         S_actual = np.asarray(S_true.copy())
#     elif (nodeID < 2*timeShift):
#         S_actual = np.asarray(S_true2.copy())
#     else:
#         S_actual = np.asarray(S_true3.copy())
        
#     D = np.where(S_est != 0)[0].shape[0]#len(numpy.where(S_est == 0)[0])
#     T = np.where(S_actual != 0)[0].shape[0]
#     TandD = float(np.where(np.logical_and(S_actual,S_est) == True)[0].shape[0])
#     P = TandD/D
#     R = TandD/T
# #    print 'D = ', D, 'T = ',T, 'TandD = ',TandD, 'P = ',P, 'R = ', R, (2* P*R/(P+R))
# #    print 'S_actual = ', S_actual
# #    print '\nS_est = ', S_est
# #    print numpy.where(numpy.logical_and(S_actual,S_est) == True)[0].shape[0]
# #    e1 = e1 + np.linalg.norm(S_actual - np.matrix(S_est), 'fro')
# #    e2 = e2 + 2* P*R/(P+R)
#     e1 = alg.norm(S_actual - np.matrix(S_est), 'fro')
#     e2 = 2* P*R/(P+R)
#     e1_set.append(e1)
#     e2_set.append(e2)
# pl.figure(1)
# pl.subplot(311)
# pl.plot(range(timesteps), e1_set)
# pl.subplot(312)
# pl.plot(range(timesteps), e2_set)
# pl.show()

#        FroError.append(e1)
#        Score.append(e2)
#        pl.plot(FroError)
#        pl.plot(Score)
#index1 = np.argmin(FroError)
#index2 = np.argmax(Score)
#index11 = index1/set_length
#index12 = index1 - (index11)*set_length
#index21 = index2/set_length
#index22 = index2 - (index21)*set_length
#print index1, index11, index12, index2, index21, index22
#print 'alpha = ', alpha_set[index11], ' beta = ', beta_set[index12], ' FroError = ', FroError[index1]
#print '\nalpha = ', alpha_set[index21], ' beta = ', beta_set[index22], ' Score = ', Score[index2]
#Fro_error = numpy.reshape(FroError,(set_length, set_length))
#Score = numpy.reshape(Score,(set_length, set_length))
#Y, X =  numpy.meshgrid(alpha_set, beta_set)
#fig1 = pl.contourf(X, Y, Score)
#pl.ylabel(r'$\alpha$')
#pl.xlabel(r'$\beta$')
#pl.colorbar()
#pl.title('Score')
#pl.savefig('fig1')
#pl.close()
#fig2 = pl.contourf(X, Y, Fro_error)
#pl.ylabel(r'$\alpha$')
#pl.xlabel(r'$\beta$')
#pl.title('frobenious error')
#pl.colorbar()
#pl.savefig('fig2')
#pl.close()
#pl.show()
#        #pl.figure(1)
#        #pl.subplot(311)
#        #pl.plot(range(timesteps), FroError)
#        #pl.subplot(312)
#        #pl.plot(range(timesteps), Score)
#        #pl.show()
#np.savez('error', size = size, timesteps = timesteps, alpha_set = alpha_set, beta_set = beta_set, S_true = S_true, S_true2 = S_true2, Fro_error = Fro_error, Score = Score )
#np.savez('alpha_beta', size = size, timesteps = timesteps, alpha_fro = alpha_set[index11], beta_fro = beta_set[index11], alpha_score = alpha_set[index21], beta_score = beta_set[index22])