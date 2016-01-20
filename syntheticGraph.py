# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 14:47:29 2016

@author: Youngsuk Park
"""


from inferGraph import *
import numpy as np
import numpy.linalg as alg
import scipy as spy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as pl
import matplotlib.animation as an
import time


def genCovariace(size): 
    MaxIter = 1e+6
    S = np.zeros((size,size))
    itn = 0
    while(alg.det(S) <= 1e-3 and itn < MaxIter):
        itn = itn + 1
        #print int(numpy.log2(size))*size
        G6 = GenRndGnm(PUNGraph, size, int((size*(size-1))*0.05))
        #G6 = snap.GenRndGnm(snap.PUNGraph, 5, 5)
        S = np.zeros((size,size))
        for EI in G6.Edges():
            S[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 #  np.random.rand(1) #2*(numpy.random.rand(1)-0.5)
        #    S[EI.GetSrcNId(), EI.GetDstNId()] =  2*(np.random.rand(1)-0.5)
    #        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
        #print S, S.max(axis = 1)
        #S =  S + S.T + np.diag( S.max(axis = 1) )#S.max()*numpy.matrix(numpy.eye(size))
        S =  S + S.T + S.max()*np.matrix(np.eye(size))
    if itn == MaxIter:
        print 'fail to find an invertible sparse inverse covariance matrix'
    S = np.asarray(S)
    return S
    
def indicesOfExtremeValue(arr, set_length, choice):
    if (choice == 'max'):
        index = np.argmax(arr)
    elif (choice == 'min'):
        index = np.argmin(arr)
    else:
        print 'invalid argument, choose max or min'
            
    index_x = index/set_length
    index_y = index - (index_x)*set_length
    return index, index_x, index_y
# Problem parameters
size = 10   
timesteps = 100 #size/2
samplesPerStep = 200#int(np.log2(size))
numberOfCov = 2
timeShift = int(np.ceil(float(timesteps)/numberOfCov)) #Number of steps till new covariance matrix appears
eps = 1e-3

# Optimization parameters
#alpha = 0.1 # Lasso parameter
#beta = 1 # Weight between basis nodes
# 0.48 and 2.2
set_length = 26
if set_length == 1:
    alpha_set = [0.1]
    beta_set  = [1.2] 
else:
    alpha_set = np.linspace(0,2,set_length)
    beta_set = np.linspace(0,4,set_length)
    
e1_set = []
e2_set = []
e3_set = []
e4_set = []
FroError = []
Score = []
AIC = []
FroThetaDiff = []

# Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)
np.random.seed(2)
#numpy.set_printoptions(suppress=True, precision = 3, threshold = 10)


S_set = []
Cov_set = []
for i in range(numberOfCov):
    S_true = genCovariace(size)
    S_set.append(S_true)
    Cov_set.append(alg.inv(S_true))
    print S_true
if numberOfCov > 1:
    numberOfDiff = float(np.where(np.logical_and(S_set[0],S_set[1]) == True)[0].shape[0])
    print numberOfDiff
#ind_zero = spy.sparse.rand(size,size,0.25).todense()
#S_true = (ind_zero + ind_zero.T)/2 + ind_zero.max()*np.matrix(np.eye(size))
#Cov = alg.inv(S_true) # Covariance matrix #1
#20
#ind_zero2 = spy.sparse.rand(size,size,0.25).todense()
#S_true2 = (ind_zero2 + ind_zero2.T)/2 + ind_zero2.max()*np.matrix(np.eye(size))
#Cov2 = alg.inv(S_true2) # second covariance matrix

sample_set = []
for alpha in alpha_set:
    for beta in beta_set:
        print alpha, beta
        gvx = TGraphVX()
        sample_set = []
        for i in range(timesteps):
        
            # Generate random samples
            x_samples = 0
            j = i/timeShift
            x_samples = np.random.multivariate_normal(np.zeros(size), Cov_set[j], samplesPerStep).T
            empCov = np.cov(x_samples)
            sample_set.append(np.asarray(empCov))
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
#                edge_obj = beta*norm(currVar['S'] - prevVar['S'],1) # one norm penalty function
                edge_obj = beta*norm(currVar['S'] - prevVar['S'],2) # two norm penalty function
                gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)
        
            #Add rake nodes, edges
            gvx.AddNode(n_id + timesteps)
            gvx.AddEdge(n_id, n_id+timesteps, Objective=alpha*norm(S,1))
        
        
        t = time.time()
        gvx.Solve()
#        gvx.Solve( MaxIters = 100)
        #gvx.Solve( NumProcessors = 1, MaxIters = 3)
        end = time.time() - t
        if set_length == 1:
            print end
        print end
        # t = time.time()
        #gvx.Solve(UseADMM=False)
        # end2 = time.time() - t
        # print "TIMES", end, end2
        
        #Plot results
        # fig = pl.figure()
        # ax1 = fig.add_subplot(2,1,1)
        # ax2 = fig.add_subplot(2,1,2)
        # ims = []
        
        
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        e1sum = 0
        e2sum = 0
        e3sum = 0
        e4sum = 0
        S_previous = np.zeros((size,size))
        for nodeID in range(timesteps):
            val = gvx.GetNodeValue(nodeID,'S')
            S_est = np.zeros((size,size))
            S_est[np.triu_indices(size)] = val #S_est matrix convert semidefinite (n(n-1) entries) to a full semidefinite matrix
            temp = S_est.diagonal()
            ind = (S_est<eps)&(S_est>-eps)
            S_est[ind] = 0
            S_est = np.asarray((S_est + S_est.T) - np.diag(temp))
            if nodeID == 0:
                S_previous = S_est
            #Get actual InvCov
            j = nodeID/timeShift
            S_actual = S_set[j]
#            print S_actual
            
            
            D = np.where(S_est != 0)[0].shape[0]#len(numpy.where(S_est == 0)[0])
            T = np.where(S_actual != 0)[0].shape[0]
            TandD = float(np.where(np.logical_and(S_actual,S_est) == True)[0].shape[0])
            P = TandD/D
            R = TandD/T
            K = float(np.where(np.logical_and((S_est>0) != (S_previous>0), S_est>0) == True)[0].shape[0])
#            K = float(np.where(np.logical_and(S_est, S_previous) == True)[0].shape[0])
            e1.append( alg.norm(S_actual - np.matrix(S_est), 'fro'))
#            if (nodeID >timeShift -3) and (nodeID < timeShift + 3):
#                print 'nodeID = ', nodeID
#                print S_actual,'\n', S_est
#                print 'D = ',D,'T = ', T,'TandD = ', TandD,'K = ', K,'P = ', P,'R = ', R,'Score = ', 2* P*R/(P+R)
#            e1.append(K)
            e2.append(R)
#            e2.append(2* P*R/(P+R))
            e3.append(-np.log(alg.det(S_est)) + np.trace(np.dot(S_est, sample_set[nodeID])) + K)
            e4.append(alg.norm(S_est -  S_previous, 'fro'))
                
            e1sum = e1sum + alg.norm(S_actual - np.matrix(S_est), 'fro')
            e2sum = e2sum + 2* P*R/(P+R)    
            e3sum = e3sum -np.log(alg.det(S_est)) + np.trace(np.dot(S_est, sample_set[nodeID])) + float(np.where(np.logical_and(np.logical_and(S_est, S_previous), S_est > 0 ) == True)[0].shape[0])
            e4sum = e4sum + alg.norm(S_est -  S_previous, 'fro')
            S_previous = S_est
            
        e1_set.append(e1)
        e2_set.append(e2)
        e3_set.append(e3)
        e4_set.append(e4)
        FroError.append(e1sum)
        Score.append(e2sum)
        AIC.append(e3sum)
        FroThetaDiff.append(e4sum)
        
index1, index11, index12 = indicesOfExtremeValue(FroError, set_length, 'min')
index2, index21, index22 = indicesOfExtremeValue(Score, set_length, 'max')
index3, index31, index32 = indicesOfExtremeValue(AIC, set_length, 'min')
index4, index41, index42 = indicesOfExtremeValue(FroThetaDiff, set_length, 'min')      

ind = index3
alpha = alpha_set[index31],
beta =  beta_set[index32]
try:
    pl.subplot(511)
    pl.plot(range(timesteps), e1_set[ind])
    pl.ylabel(r'$S_{true} - S_{est}$')
    pl.subplot(512)
    pl.plot(range(timesteps), e2_set[ind])
    pl.ylabel('Score')
    pl.subplot(513)
    pl.plot(range(timesteps), e3_set[ind])
    pl.ylabel('AIC')
    pl.subplot(514)
    pl.semilogy(range(timesteps), e4_set[ind])
    pl.ylabel(r'$S_i - S_{i+1}$')
    pl.title(r'($\alpha$, $\beta$ ) = (%s, %s)'%(alpha, beta))
    pl.savefig('MeasurePlot')
#    pl.show()
    print '\nSuceed to save MeasurePlot'
except:
    print 'fail to save plots'
if set_length > 1:
    #print index1, index11, index12, index2, index21, index22
    print 'alpha = ', alpha_set[index11], ' beta = ', beta_set[index12], ' FroError = ', FroError[index1]
    print 'alpha = ', alpha_set[index21], ' beta = ', beta_set[index22], ' Score = ', Score[index2]
    print 'alpha = ', alpha_set[index31], ' beta = ', beta_set[index32], ' AIC = ', AIC[index3]
    print 'alpha = ', alpha_set[index41], ' beta = ', beta_set[index42], ' FroThetaDiff = ', FroThetaDiff[index4]
    
    Fro_error = numpy.reshape(FroError,(set_length, set_length))
    Score = numpy.reshape(Score,(set_length, set_length))
    AIC = numpy.reshape(AIC,(set_length, set_length))
    FroThetaDiff = numpy.reshape(FroThetaDiff,(set_length, set_length))
    np.savez('ErrorMeasure%s'%(set_length), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
             alpha_set = alpha_set, beta_set = beta_set, S_set = S_set, 
             Fro_error = Fro_error, Score = Score, AIC =  AIC, FroThetaDiff=  FroThetaDiff,
             e1_set = e1_set, e2_set = e2_set, e3_set = e3_set, e4_set = e4_set)
    np.savez('OptPars%s'%(set_length), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
             alpha_fro = alpha_set[index11], beta_fro = beta_set[index11], 
             alpha_score = alpha_set[index21], beta_score = beta_set[index22], 
             alpha_AIC = alpha_set[index31], beta_AIC = beta_set[index31], 
             alpha_theta = alpha_set[index41], beta_theta = beta_set[index42])
    
    Y, X =  np.meshgrid(alpha_set, beta_set)
#    pl.figure(1)
    try:
        pl.subplot(221)
        pl.contourf(X, Y, Score)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.colorbar()
        pl.title('Score')
        pl.subplot(222)
        pl.contourf(X, Y, Fro_error)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title(r'$S_{true} - S_{est}$')
        pl.colorbar()
        pl.subplot(223)
        pl.contourf(X, Y, AIC)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title('AIC')
        pl.colorbar()
        pl.subplot(224)
        pl.contourf(X, Y, FroThetaDiff)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title(r'$S_i - S_{i+1}$')
        pl.colorbar()
        pl.savefig('GridGraph%s'%(set_length))
        pl.show()
        print ('\nSuceed to save GridGraph%s'%(set_length))
    except:
        print 'fail to save graph'
print '\nEnd'
