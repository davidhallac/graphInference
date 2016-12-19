# -*- coding: utf-8 -*-
"""
Code to generate data and obtain results for the L2 synthetic graph from part 5a
"""
      
import numpy as np
import numpy.linalg as alg
import scipy as spy

#import matplotlib as mpl
import matplotlib.pylab as pl
import time
 

def timing_set(center, samplesPerStep_left, count_left, samplesPerStep_right, count_right ):
    time_set = []
    count_left = min(count_left, center/samplesPerStep_left)
    print 'left timesteps: = ', count_left
    start = max(center- samplesPerStep_left*(count_left), 0)
    for i in range(count_left):
        time_interval = [start, start + samplesPerStep_left -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_left
    count_right = min(count_right, 245/samplesPerStep_left)
    print 'right timesteps: = ', count_right
    for i in range(count_right):
        time_interval = [start, start + samplesPerStep_right -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_right
    return time_set
    
# Problem parameters
size    = 10
timestamps = 100
samplesPerStep = 10
numberOfCov = 4
timeShift = int(np.ceil(float(timestamps)/numberOfCov)) #Number of steps till new covariance matrix appears
eps     = 3e-3
epsAbs  = 1e-3
epsRel  = 1e-3

# Choose a penalty function
# 1: l1, 2: l2, 3: laplacian, 4: l-inf, 5: perturbation node penalty
index_penalty = 2


# Choose the number of alpha/beta point 
setLength   = 1      # setLength = 1 indicates of using a fixed alpha/beta

# Parameters for l2 penalty


# Parameters for perturbed node penalty
#aa = 0.28
#bb = 5

# Covariance matrix parameters
dataType= 'Syn'
cov_mode = index_penalty # 1,2,4: normal cov, 3: cov for laplacian, 5: perturbation 
low     = 0.3
upper   = 0.6
compare = True



if dataType == 'Stock':
    samplesPerStep = 5
    time_set = timing_set(101, samplesPerStep, 6, samplesPerStep, 8)


# Kernel parameters
kernel_width = 1 # kernel width for naive method and for TVGL under kernel usage
# this kernel width is currently dummy because it is automatically decide!!

kernel_sigma = 10 # kernel sigma for naive method and for TVGL under kernel usage
kernel_use = False # True/False:  use/not use kernel for TVGL


if setLength == 1:
#    alpha_set = [0.3]
#    if kernel_use == False:
#        beta_set = [10.0]
#    else:
#        print 'kernel = ', kernel_use, 'beta is kernel width'
#        beta_set  = [7.0] # kernel_width
    if dataType == 'Syn': # Parameters for penalty function
        alpha_set   = [0.3]
        beta_set    = [7]
    elif dataType == 'Stock':
        alpha_set   = [0.27] # apple case and flash crash
        beta_set    = [10]  # apple case
        
else:
    compare     = False
    alpha_set   = np.linspace(0.2, 1 , setLength)
    beta_set    = np.logspace(0, 1.3, setLength)
#    if kernel_use == False:
#        beta_set = np.logspace(0, 1.3, setLength)
#    else:
#        print 'kernel = ', kernel_use, 'beta is kernel width'
#        interval = min(timeShift/setLength, 5)
#        beta_set = np.arange(1,setLength * interval, interval)
         


if index_penalty == 1:
    print 'Use l-1 penalty function'
    from inferGraph1 import *
elif index_penalty == 2:
    print 'Use l-2 penalty function'
    from inferGraph2 import *
elif index_penalty == 3:
    print 'Use laplacian penalty function'
    from inferGraph3 import *
elif index_penalty == 4:
    print 'Use l-inf penalty function'
    from inferGraph4 import *
else:
    print 'Use perturbation node penalty function'
    from inferGraph5 import *


#--------------------------------------- Define private functions ------------------------------------------
def genCovariace(size): 
    MaxIter = 1e+6
    S = np.zeros((size,size))
    itn = 0
    while(alg.det(S) <= 1e-3 and itn < MaxIter):
        itn = itn + 1
        #print int(numpy.log2(size))*size
        G6 = GenRndGnm(PUNGraph, size, int((size*(size-1))*0.05))
        S = np.zeros((size,size))
        for EI in G6.Edges():
            S[EI.GetSrcNId(), EI.GetDstNId()] = 0.6
        S =  S + S.T + S.max()*np.matrix(np.eye(size))
    if itn == MaxIter:
        print 'fail to find an invertible sparse inverse covariance matrix'
    S = np.asarray(S)
    return S
    
def genInvCov(size, low = 0 , upper = 0.6, portion = 0.05):
    S = np.zeros((size,size))
    G = GenRndGnm(PUNGraph, size, int((size*(size-1))*portion))
    for EI in G.Edges():
        value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])  
#        print value
        S[EI.GetSrcNId(), EI.GetDstNId()] = value
    S = S + S.T
    vals = alg.eigvalsh(S)
    S = S + (0.1 - vals[0])*np.identity(size)
    return np.matrix(S)
    
def genMulCov(size, numberOfCov, low, upper, mode, portion = 0.05):
    S_set = []   
    Cov_set = []
    minEVal_set = [] 
    m = size/3
    mm = m/2
#    print m, mm
    S_init = np.zeros((size,size))
    for k in range(numberOfCov):
        S = np.zeros((size,size))
        if k == 0:
            S = genInvCov(size, low, upper, portion)
            if mode == 5:      
                ind_zero = np.where(spy.sparse.rand(m, size-m, 0.5).todense() == 0)
                value = np.multiply((np.random.randint(2, size = (m, size -m)) - 0.5)*2,(low + (upper - low)*np.random.rand(m,size -m)))
                value[ind_zero] = 0
                hub = value
                S[:m, m:size] = hub
                S[m:size, :m] = hub.T        
                minEVal_set.append(alg.eigvalsh(S)[0])
            S_init = S
        elif mode == 3: #'laplacian'
            ind1 = range(m)
            ind2 = np.random.permutation(m)
            S = np.copy(S_init)
            S[ind1, :] = S[ind2, :]            
            S[:, ind1] = S[:, ind2]
        elif mode  == 5: #'perturbation'
            S = np.copy(S_init)
            ind_zero = np.where(spy.sparse.rand(mm, size-mm, 0.5).todense() == 0)
            pert = np.multiply((np.random.randint(2, size = (mm, size -mm)) - 0.5)*2,(low + (upper - low)*np.random.rand(mm,size -mm)))
            pert[ind_zero] = 0 
            S[:mm, mm:size] = pert
            S[mm:size, :mm] = pert.T
            minEVal_set.append(alg.eigvalsh(S)[0])
        else:
#            print 'Activate normal mode'
            S = genInvCov(size, low, upper, portion)
        S_set.append(S)
    
    for k in range(numberOfCov):
        if mode == 5:
            S_set[k] = S_set[k] + (0.1 - min(minEVal_set))*np.identity(size)
        Cov_set.append(alg.inv(S_set[k]))
    return S_set, Cov_set   
     
    
def genSampleSet(Cov_set, samplesPerStep, timestamps, timeShift):
    sample_set = []
    for i in range(timestamps):
        j       = i/timeShift # where to change the covariance matrix
        samples = np.random.multivariate_normal(np.zeros(size), Cov_set[j], samplesPerStep).T
        sample_set.append(samples)
    return sample_set


def genEmpCov(sample_set, kernel_use, kernel_width = 1, kernel_sigma = 1):
    timestamps      = sample_set.__len__()
    samplesPerStep  = sample_set[0].shape[1]
    if kernel_use:
        kernel_width = np.ceil(np.power(timestamps, 1.0/3 ))
        kernel_sigma = float(kernel_sigma)
    else:
        kernel_width = 1
    
#    print 'kernel_width is', kernel_width
    empCov_set = []
    for i in range(timestamps):
        w_sum = 0
        empCov = 0
        for j in range( int(max(0,i + 1 - kernel_width )),i + 1 ):
            w      = np.exp( -np.square(i- j) / kernel_sigma )
#            print w,
            m_tile = np.tile( np.mean(sample_set[j], axis = 1), (samplesPerStep,1) ).T
            X      = sample_set[j]- m_tile
            empCov = empCov + w*np.dot(X, X.T)/samplesPerStep
            w_sum  = w_sum + w
        empCov = empCov/w_sum
#        print 'w_sum is ', w_sum
        empCov_set.append(np.asarray(empCov))
    return empCov_set
    
def getStocks(time_set, stock_list, data):
    timesteps = time_set.__len__()
    sample_set = []
    empCov_set = []
    stock_data = np.genfromtxt('finance.csv', delimiter=',')
#    print stock_data
    size = stock_list.__len__()
#    print 'timesteps = ',timestpes
    for i in range(timesteps):
        time_interval = time_set[i] 
#        print time_interval
        sample_data = stock_data[time_interval[0]:time_interval[1], stock_list].T
        sample_data_set = sample_set.append(sample_data)
        empCov_set.append(genEmpCov(sample_data))
    return size, timesteps, sample_data_set, empCov_set
    

def indicesOfExtremeValue(arr, setLength, choice):
    if (choice == 'max'):
        index = np.argmax(arr)
    elif (choice == 'min'):
        index = np.argmin(arr)
    else:
        print 'invalid argument, choose max or min'
            
    index_x = index/setLength
    index_y = index - (index_x)*setLength
    return index, index_x, index_y
    
def upper2Full(a, eps = 0):
    ind = (a<eps)&(a>-eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))             
    return A   
    
def solveProblem(gvx, index_penalty, alpha, beta, empCov_set, epsAbs = 1e-4, epsRel = 1e-4):
    # Solve the problem via SnapVX, being passed by empirical covariance matrices
    timestamps = empCov_set.__len__()
#    print 'here1'
    for i in range(timestamps):
        #Add Node, edge to previous timestamp
        empCov = empCov_set[i] 
        n_id = i
        S = semidefinite(size,name='S')
        obj = -log_det(S) + trace(empCov*S) #+ alpha*norm(S,1)
        gvx.AddNode(n_id, obj)
        
        if (i > 0): #Add edge to previous timestamp
            prev_Nid = n_id - 1
            currVar = gvx.GetNodeVariables(n_id)
            prevVar = gvx.GetNodeVariables(prev_Nid)
            if index_penalty == 1 or index_penalty == 2:
                edge_obj = beta*norm(currVar['S'] - prevVar['S'],index_penalty) # norm 1, 2 penalty function
            elif index_penalty == 3:
                edge_obj = beta*square(norm(currVar['S'] - prevVar['S'],index_penalty)) # laplacian penalty function
            elif index_penalty == 4:
                edge_obj = beta*norm(currVar['S'] - prevVar['S'], np.inf) # norm 1, 2 penalty function
            else:
                edge_obj = beta*norm(currVar['S'] - prevVar['S'],index_penalty)
            gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)
        
        #Add rake nodes, edges
        gvx.AddNode(n_id + timestamps)
        gvx.AddEdge(n_id, n_id + timestamps, Objective=alpha*norm(S,1))
#    print 'here2'    
    t = time.time()
    gvx.Solve( EpsAbs=epsAbs, EpsRel=epsRel )
#    gvx.Solve( EpsAbs=epsAbs, EpsRel=epsRel ,NumProcessors = 1,  Verbose = True)
    end = time.time() - t
    print 'time span = ',end
    return gvx
    
    
    # [12/18/16, 2:00:28 PM] David Hallac: numProcessors
def genGraph(S_actual, S_est, S_previous, empCov_set, nodeID, e1, e2, e3, e4, display = False):
    D   = np.where(S_est != 0)[0].shape[0]
    T   = np.where(S_actual != 0)[0].shape[0]
    TandD = float(np.where(np.logical_and(S_actual,S_est) == True)[0].shape[0])
    P   = TandD/D
    R   = TandD/T
    
    offDiagDiff = S_actual - S_est
    offDiagDiff = offDiagDiff - np.diag(np.diag(offDiagDiff))
    S_diff = (S_est - S_previous)  
    S_diff = S_diff - np.diag(np.diag(S_diff))
    ind = (S_diff < 1e-2) & (S_diff > - 1e-2)
    S_diff[ind] = 0    
    K = np.count_nonzero(S_diff)
    e1.append( alg.norm(offDiagDiff, 'fro'))
    e2.append(2* P*R/(P+R))
    
    
    K = float(np.where(np.logical_and((S_est>0) != (S_previous>0), S_est>0) == True)[0].shape[0])
    e3.append(-np.log(alg.det(S_est)) + np.trace(np.dot(S_est, empCov_set[nodeID])) + K)
    e4.append(alg.norm(S_est -  S_previous, 'fro'))
    
    display = False
    if display == True:
        if (nodeID >timeShift -10) and (nodeID < timeShift + 10):
            print 'nodeID = ', nodeID
            print 'S_true = ', S_actual,'\nS_est', S_est
#            print 'S_error = ',S_actual - S_est, '\n its Fro error = ', alg.norm(S_actual - S_est, 'fro')
            print 'D = ',D,'T = ', T,'TandD = ', TandD,'K = ', K,'P = ', P,'R = ', R,'Score = ', 2* P*R/(P+R)
            
    return e1, e2, e3, e4
#--------------------------------------------- End Defining functions -------------------------------------- 


# Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)
np.random.seed(1)
np.set_printoptions(suppress=True, precision = 3)


# Print the data information and generate the set of underlying covariance matrices
S_set   = []
Cov_set = []
sentence = ''
if dataType == 'Syn':
    if cov_mode == 0:
        for i in range(numberOfCov):
            S_true = genCovariace(size)
            S_set.append(S_true)
            Cov_set.append(alg.inv(S_true))
        sentence = 'Synthetic data' + '_fixed'
    else:
        S_set, Cov_set = genMulCov(size, numberOfCov, low, upper, cov_mode)
        sentence = dataType + 'with penalty/cov mode %s'%(cov_mode)
    sentence = sentence + '(for the size of %s)'%(size)
else:
    size, timesteps, sample_set_stock, empCov_set_stock = getStocks(time_set, stock_list,'finance.csv')
    sentence = data_type + '(%s)'%(size)
print '********',sentence,'********'
#print S_set
   
    
    
sample_set = [] 
empCov_set = []
if dataType == 'Stock':
#    print 'analyze stock data'  
    size, timesteps, sample_set_stock, empCov_set_stock = getStocks(time_set, stock_list,'finance.csv')
    empCov_set_naive = empCov_set_stock
#    sample_set_naive = sample_set_stock
else:
    sample_set = genSampleSet(Cov_set, samplesPerStep, timestamps, timeShift)
    empCov_set = genEmpCov(sample_set, kernel_use, kernel_width, kernel_sigma)
    empCov_set_kernel = genEmpCov(sample_set, True, kernel_width, kernel_sigma)
    empCov_set_static = genEmpCov(sample_set, False, kernel_width, kernel_sigma)
#for i in range(5):
#    print empCov_set_static[i], empCov_set_kernel[i]
    
    
e1_set = []
e2_set = []
e3_set = []
e4_set = []
e1_kernel = []
e2_kernel = []   
e3_kernel = []   
e4_kernel = []   
e5_kernel = []   
e1_static = []
e2_static = []   
e3_static = []   
e4_static = []   
e5_static = []   

FroError = []
Score = []
AIC = []
FroThetaDiff = []
print 'number of samples per time: ', samplesPerStep

#print empCov_set
for alpha in alpha_set:
    for beta in beta_set:
        print '--------------------- alpha = %s, beta = %s --------------------'%(alpha, beta)
        gvx = TGraphVX()   
        gvx_kernel = TGraphVX()
        gvx_static = TGraphVX()

#        print 'solve gvx'
        gvx = solveProblem(gvx, index_penalty, alpha, beta, empCov_set, epsAbs, epsRel)
        if setLength == 1 and compare == True:
#            print 'solve naive gvx'
            gvx_kernel = solveProblem(gvx_kernel, index_penalty, alpha, 0, empCov_set_kernel, epsAbs, epsRel) 
            gvx_static = solveProblem(gvx_static, index_penalty, alpha, 0, empCov_set_static, epsAbs, epsRel) 

        e1 = []
        e2 = []
        e3 = []
        e4 = []
        S_previous = np.zeros((size,size))
        S_kernel_previous = np.zeros((size,size))
        S_static_previous = np.zeros((size,size))

        for nodeID in range(timestamps):
            val = gvx.GetNodeValue(nodeID,'S')
            S_est = upper2Full(val, eps)
            if nodeID == 0:
                S_previous = S_est
            if dataType == 'Syn':
                #Get actual InvCov
                j = nodeID/timeShift
                S_actual = np.asarray(S_set[j])
    #            print 'S_actual=', S_actual
            else:
                S_actual = np.identity(size)
            
            e1, e2, e3, e4 = genGraph(S_actual, S_est, S_previous, empCov_set, nodeID, e1, e2, e3, e4, False)
            
            S_previous = S_est
            
            if setLength == 1 and compare == True:               
                val_kernel = gvx_kernel.GetNodeValue(nodeID,'S')
                S_kernel = upper2Full(val_kernel, eps)
                e1_kernel, e2_kernel, e3_kernel, e4_kernel = genGraph(S_actual, S_kernel, S_kernel_previous, empCov_set_kernel, nodeID, e1_kernel, e2_kernel, e3_kernel, e4_kernel, False)
                S_kernel_previous =  S_kernel
                          
                val_static = gvx_static.GetNodeValue(nodeID,'S')
                S_static = upper2Full(val_static, eps)
                e1_static, e2_static, e3_static, e4_static = genGraph(S_actual, S_static, S_static_previous, empCov_set_static, nodeID, e1_static, e2_static, e3_static, e4_static, False)
                S_static_previous =  S_static
            
            
        e1_set.append(e1)
        e2_set.append(e2)
        e3_set.append(e3)
        e4_set.append(e4)
#        print e4
        FroError.append(sum(e1))
        Score.append(sum(e2))
        AIC.append(sum(e3))
        FroThetaDiff.append(sum(e4))
        
index1, index11, index12 = indicesOfExtremeValue(FroError, setLength, 'min')
index2, index21, index22 = indicesOfExtremeValue(Score, setLength, 'max')
index3, index31, index32 = indicesOfExtremeValue(AIC, setLength, 'min')
index4, index41, index42 = indicesOfExtremeValue(FroThetaDiff, setLength, 'min')      

ind = index3
alpha = alpha_set[index31]
beta =  beta_set[index32]
#try:
#print alpha
x =  range(1,timestamps+1)  
#np.savetxt('x.csv'  , x)
##np.savetxt('alpha.csv', alpha)    
##np.savetxt('beta.csv' , beta)  
#np.savetxt('e1.csv' , e1_set[ind])    
#np.savetxt('e2.csv' , e2_set[ind])    
#np.savetxt('e4.csv' , e4_set[ind])     
#np.savetxt('e11.csv', e1_kernel)    
#np.savetxt('e21.csv', e2_kernel)    
#np.savetxt('e41.csv', e4_kernel)    

    
if dataType == 'Syn':
    ax1 = pl.subplot(311)    
    pl.title(r'Results for Global Shift with $\ell_2$ Penalty')
    pl.plot(x, e1_set[ind])
    pl.yticks([0.8,1.0,1.2,1.4])
    pl.axvline(x=51,color='r',ls='dashed')
    pl.ylim([0.65,1.45])
    pl.ylabel('Abs. Error')
    ax1.set_xticklabels([])
    
    ax2 = pl.subplot(312)
    pl.plot(x, e2_set[ind])
    pl.yticks([0.4,0.6,0.8,1.0])
    pl.axvline(x=51,color='r',ls='dashed')
    pl.ylim([0.3,1.0])
    pl.ylabel(r'$F_1$')
    ax2.set_xticklabels([])
    
else:
    pl.subplot(311)    
    pl.plot(x, e1_set[ind])
    pl.title(r'%s, $n_t$ = %s, ($\lambda$, $\beta$) = (%s, %s)'%(sentence, samplesPerStep, alpha, beta))

ax3 = pl.subplot(313)
david1, = pl.semilogy(x, e4_set[ind])
pl.axvline(x=51,color='r',ls='dashed')
pl.ylabel('Temp. Dev.')
pl.xlabel('Timestamp')
pl.rcParams.update({'font.size':14})

print '\nave_PN:', np.mean(e1_set[ind][:49]),  np.mean(e1_set[ind][51:]), np.mean(e1_set[ind])
print '\nave_PN:', np.mean(e2_set[ind][:49]),  np.mean(e2_set[ind][51:]), np.mean(e2_set[ind])
print '\nave_PN:', np.mean(e4_set[ind][:49]),  np.mean(e4_set[ind][51:]), np.mean(e4_set[ind])
if setLength == 1 and compare == True:
    pl.subplot(311)     
    pl.plot(x, e1_kernel, label = 'kernel')
    pl.plot(x, e1_static, label = 'static')
    pl.subplot(312)
    pl.plot(x, e2_kernel)    
    pl.plot(x, e2_static)

    pl.subplot(313)
    david2, = pl.semilogy(x, e4_kernel)
    david2, = pl.semilogy(x, e4_static)
    pl.rc('legend',**{'fontsize':14})
    david3 = pl.legend([david1,david2],['TVGL','Baseline'], ncol=2, loc=7, bbox_to_anchor=(1,0.57), columnspacing=0.4) 
    david3.draw_frame(False)  
    print '--------- Kernel method -------'
    print 'Abs err  :', np.mean(e1_kernel)
    print 'F1 score :', np.mean(e2_kernel)
    print 'Temp Dev :', np.mean(e4_kernel)
    
    print '--------- Static method -------'
    print 'Abs err  :', np.mean(e1_static)
    print 'F1 score :', np.mean(e2_static)
    print 'Temp Dev :', np.mean(e4_static)
Data_type = dataType + '%s'%(cov_mode) + '%s'%(samplesPerStep)
pl.savefig(Data_type)
pl.savefig(Data_type+'.eps', format = 'eps', bbox_inches = 'tight', dpi = 1000)
pl.show()
#if setLength > 1:
#    #print index1, index11, index12, index2, index21, index22
#    print 'alpha = ', alpha_set[index11], ' beta = ', beta_set[index12], ' FroError = ', FroError[index1]
#    print 'alpha = ', alpha_set[index21], ' beta = ', beta_set[index22], ' Score = ', Score[index2]
#    print 'alpha = ', alpha_set[index31], ' beta = ', beta_set[index32], ' AIC = ', AIC[index3]
#    print 'alpha = ', alpha_set[index41], ' beta = ', beta_set[index42], ' FroThetaDiff = ', FroThetaDiff[index4]
#    
#    Fro_error = np.reshape(FroError,(setLength, setLength))
#    Score = np.reshape(Score,(setLength, setLength))
#    AIC = np.reshape(AIC,(setLength, setLength))
#    FroThetaDiff = np.reshape(FroThetaDiff,(setLength, setLength))
#    np.savez('ErrorMeasure%s'%(setLength), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
#             alpha_set = alpha_set, beta_set = beta_set, S_set = S_set, 
#             Fro_error = Fro_error, Score = Score, AIC =  AIC, FroThetaDiff=  FroThetaDiff,
#             e1_set = e1_set, e2_set = e2_set, e3_set = e3_set, e4_set = e4_set)
#    np.savez('OptPars%s'%(setLength), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
#             alpha_fro = alpha_set[index11], beta_fro = beta_set[index11], 
#             alpha_score = alpha_set[index21], beta_score = beta_set[index22], 
#             alpha_AIC = alpha_set[index31], beta_AIC = beta_set[index31], 
#             alpha_theta = alpha_set[index41], beta_theta = beta_set[index42])
#    
#    Y, X =  np.meshgrid(alpha_set, beta_set)
##    pl.figure(1)
#    try:
#        pl.subplot(221)
#        pl.contourf(X, Y, Score)
#        pl.ylabel(r'$\alpha$')
#        pl.xlabel(r'$\beta$')
#        pl.colorbar()
#        pl.title('Score')
#        pl.subplot(222)
#        pl.contourf(X, Y, Fro_error)
#        pl.ylabel(r'$\alpha$')
#        pl.xlabel(r'$\beta$')
#        pl.title(r'$S_{true} - S_{est}$')
#        pl.colorbar()
#        pl.subplot(223)
#        pl.contourf(X, Y, AIC)
#        pl.ylabel(r'$\alpha$')
#        pl.xlabel(r'$\beta$')
#        pl.title('AIC')
#        pl.colorbar()
#        pl.subplot(224)
#        pl.contourf(X, Y, FroThetaDiff)
#        pl.ylabel(r'$\alpha$')
#        pl.xlabel(r'$\beta$')
#        pl.title(r'$S_i - S_{i+1}$')
#        pl.colorbar()
#        pl.savefig('GridGraph%s'%(setLength))
#        pl.show()
#        print ('\nSuceed to save GridGraph%s'%(setLength))
#    except:
#        print 'fail to save graph'
print '\nEnd'
