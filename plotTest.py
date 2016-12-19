 
import numpy as np
import numpy.linalg as alg
import scipy as spy

import matplotlib as mpl
import matplotlib.pylab as pl


alpha = np.genfromtxt('alpha.csv', delimiter = ',' ) 
beta = np.genfromtxt('beta.csv', delimiter = ',' ) 
x = np.genfromtxt('x.csv', delimiter = ',' ) 
e1 = np.genfromtxt('e1.csv', delimiter = ',' ) 
e2 = np.genfromtxt('e2.csv', delimiter = ',' ) 

e4 = np.genfromtxt('e4.csv', delimiter = ',' ) 
e11 = np.genfromtxt('e11.csv', delimiter = ',' ) 
e21 = np.genfromtxt('e21.csv', delimiter = ',' ) 

e41 = np.genfromtxt('e41.csv', delimiter = ',' ) 




setLength = 1
dataType = 'Syn'




#if dataType == 'Syn':
ax1 = pl.subplot(311)    
pl.title(r'Results for Global Shift with $\ell_2$ Penalty')
pl.plot(x, e1)
pl.yticks([0.8,1.0,1.2,1.4])
pl.axvline(x=51,color='r',ls='dashed')
pl.ylim([0.65,1.45])
pl.ylabel('Abs. Error')
ax1.set_xticklabels([])

ax2 = pl.subplot(312)
pl.plot(x, e2)
pl.yticks([0.4,0.6,0.8,1.0])
pl.axvline(x=51,color='r',ls='dashed')
pl.ylim([0.3,1.0])
pl.ylabel(r'$F_1$')
ax2.set_xticklabels([])
#    
#else:
#    pl.subplot(311)    
#    pl.plot(x, e1_set[ind])
#    pl.title(r'%s, $n_t$ = %s, ($\lambda$, $\beta$) = (%s, %s)'%(sentence, samplesPerStep, alpha, beta))

ax3 = pl.subplot(313)
david1, = pl.semilogy(x, e4)
pl.axvline(x=51,color='r',ls='dashed')
pl.ylabel('Temp. Dev.')
pl.xlabel('Timestamp')
pl.rcParams.update({'font.size':14})
#
#print '\nave_PN:', np.mean(e1_set[ind][:49]),  np.mean(e1_set[ind][51:]), np.mean(e1_set[ind])
#print '\nave_PN:', np.mean(e2_set[ind][:49]),  np.mean(e2_set[ind][51:]), np.mean(e2_set[ind])
#print '\nave_PN:', np.mean(e4_set[ind][:49]),  np.mean(e4_set[ind][51:]), np.mean(e4_set[ind])
#if setLength == 1 and compare == True:
#    pl.subplot(311)     
#    pl.plot(x, e11, label = 'zero beta')
#    pl.subplot(312)
#    pl.plot(x, e21)
#    pl.subplot(313)
#    david2, = pl.semilogy(x, e41)
#    pl.rc('legend',**{'fontsize':14})
#    david3 = pl.legend([david1,david2],['TVGL','Baseline'], ncol=2, loc=7, bbox_to_anchor=(1,0.57), columnspacing=0.4) 
#    david3.draw_frame(False)   
#    print '\nave_Naive:', np.mean(e11)
#    print '\nave_Naive:', np.mean(e21)
#    print '\nave_Naive:', np.mean(e41)
#Data_type = dataType + '%s'%(cov_mode) + '%s'%(samplesPerStep)
#pl.savefig(Data_type)
#pl.savefig(Data_type+'.eps', format = 'eps', bbox_inches = 'tight', dpi = 1000)
#pl.show()