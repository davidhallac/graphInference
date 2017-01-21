import shlex, subprocess

for a in [5]:
	for b in [2, 3, 4]:
		print 'Now analyze', ' underlying ', a, ' type of covariance with ', b, ' penalty'
		args = 'python2.7 MainSynGraph.py ' +  str(a) + ' ' + str(b) + ' > output'+str(a)+str(b)+'.txt'
		p = subprocess.Popen(args,stdout=subprocess.PIPE, shell = True)
		out,err = p.communicate()
