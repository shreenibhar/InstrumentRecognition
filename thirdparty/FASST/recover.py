import os
import numpy
import shlex
import scipy.io

proc_files = os.listdir('results')
for i in range(len(proc_files)):
	proc_files[i] = proc_files[i].split('__')[0] + '.wav'
assert len(proc_files) % 4 == 0
proc_files = list(set(proc_files))
for filer in proc_files:
	to_del = os.path.join('example_data', filer)
	to_del = shlex.quote(to_del)
	os.system('rm ' + to_del)
print('recovered %d samples.' % len(proc_files))
print('deleted the recovered samples.')
