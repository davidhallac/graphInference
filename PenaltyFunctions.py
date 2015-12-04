from cvxpy import *
import numpy as numpy
import time


numpy.random.seed(1)
numpy.set_printoptions(suppress=True, precision = 4, threshold = 5)
timeset = []
n = 10
#    n = 100
global rho 
rho = 1.2
A_1 = numpy.random.randn(n,n)
A_2 = numpy.copy(A_1)
a = numpy.random.randn(n)
A_2[0,:] = a
A_2[:,0] = a.T
A_1 = A_1 + A_1.T
A_2 = A_2 + A_2.T
print '======================================================'
print 'Parameters: n = ',n,', rho = ',rho
print 'Node perturbation matrix we want to detect: \n',A_1 -A_2
print '======================================================\n'


#1. Naive l2 penalty
t1 = time.time()
theta_1 = Variable(n,n)
theta_2 = Variable(n,n)
constraints = []
objective = Minimize( mixed_norm((theta_1 - theta_2).T,2,1) + rho/2*sum_squares(theta_1 - A_1) + rho/2*sum_squares(theta_2 - A_2) )
problem = Problem(objective, constraints)
problem.solve()
t1 = time.time() - t1
print '------------------------------------------------------'
print 'Naive l2 norm penalty:\nelapsed time = ',t1
print 'Theta1 - Theta2 = \n', (theta_1-theta_2).value


#2. Naive l2 penalty with symmetry constraints
t2 = time.time()
theta_1 = semidefinite(n)
theta_2 = semidefinite(n)
constraints = []
objective = Minimize( mixed_norm((theta_1 - theta_2).T,2,1) + rho/2*sum_squares(theta_1 - A_1) + rho/2*sum_squares(theta_2 - A_2) )
problem = Problem(objective, constraints)
problem.solve()
t2 = time.time() - t2

print '------------------------------------------------------'
print 'Naive l2 norm penalty with symmetry constraints:\nelapsed time = ',t2
print 'Theta1 - Theta2 = \n', (theta_1-theta_2).value

#3. node-perturbation penalty using cvxpy
t3 = time.time()
theta_1 = Variable(n,n)
theta_2 = Variable(n,n)
V = Variable(n,n)
W = Variable(n,n)
constraints = [(V + W) == (theta_1 - theta_2),
                V == W.T]
#constraints = [D == (theta_1 - theta_2)]
objective = Minimize( mixed_norm(V.T,2,1) + rho/2*sum_squares(theta_1 - A_1) + rho/2*sum_squares(theta_2 - A_2) )
problem = Problem(objective, constraints)
problem.solve()
t3 = time.time() - t3
print '------------------------------------------------------'
print 'Node-perturbation penalty using CVXPY:\nelapsed time = ',t3
#print 'V = \n', V.value
print 'Theta1 - Theta2 = \n',(theta_1-theta_2).value
Theta1 = (theta_1-theta_2).value

##4. node-perturbation penalty for augmented L using cvxpy
#theta_1 = Variable(n,n)
#theta_2 = Variable(n,n)
#U1 = numpy.zeros([n,n])
#U2 = numpy.zeros([n,n])
#V = Variable(n,n)
#W = Variable(n,n)
#constraints = []
#
#t = time.time()
#objective = Minimize( mixed_norm(V.T,2,1) + rho/2*sum_squares(theta_1 - A_1) + rho/2*sum_squares(theta_2 - A_2) 
#                      + rho/2*sum_squares((V + W) - (theta_1 - theta_2) + U1) + rho/2*sum_squares(V - W.T + U2) )
#problem = Problem(objective, constraints)
#problem.solve()
#t = time.time() - t
#print '------------------------------------------------------'
#print 'node-perturbation norm penalty for augmented L :\nelapsed time = ',t
##print 'D = \n', D.value
#print 'Theta1 - Theta2 = \n',(theta_1-theta_2).value


def Prox_twonorm(A, eta):
    col_norms = numpy.linalg.norm(A, axis = 0)
    Z = numpy.dot(A, numpy.diag((numpy.ones(n) - eta/col_norms)*(col_norms > eta)))
    return Z 
    
def Prox_node_penalty(A_ij, A_ji, beta, MaxIter):
    global rho
    I = numpy.identity(n)  
    U = numpy.ones([n,n])/n
    U1 = numpy.ones([n,n])/n
    U2 = numpy.ones([n,n])/n
    theta_1 = numpy.copy(U)
    theta_2 = numpy.copy(U)
    V = numpy.copy(U)
    W = numpy.copy(U)
    
    for k in range(MaxIter):
        A = ((theta_1 - theta_2 - W - U1) + (W.T - U2))/2
        eta = beta/(2*rho)
        V = Prox_twonorm(A, eta)
    
        eta = (rho/2)/rho
        C = numpy.concatenate((I,-I, I), axis = 1)        
        C = numpy.matrix(C)
        A = numpy.concatenate([(V+U2).T, A_1, A_2], axis = 0)
        D = V + U1

        Z = numpy.linalg.solve(C.T*C + eta*numpy.identity(3*n), - C.T*D + eta* A)
        W = Z[:n,:]
        theta_1 = Z[n:2*n,:]
        theta_2 = Z[2*n:,:]    
    
        U1 = U1 + ((V + W) - (theta_1 - theta_2)) 
        U2 = U2 + (V - W.T)
    return theta_1,theta_2

tt = time.time()
beta = 1
MaxIter = 15
[theta_1, theta_2] = Prox_node_penalty(A_1, A_2, beta, MaxIter)
tt = time.time() - tt
print '------------------------------------------------------'
print 'Node-perturbation penalty using ADMM :\nelapsed time = ',tt
print 'Theta1 - Theta2 = \n',(theta_1-theta_2)
print 'Using ADMM for %d by %d matrix is (%d, %d, %d) times faster ' %( n, n, t1/tt, t2/tt, t3/tt)


print numpy.linalg.norm(Theta1 - (theta_1-theta_2),'fro')/numpy.linalg.norm(Theta1,'fro')
