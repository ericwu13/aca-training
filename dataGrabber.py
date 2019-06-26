from profiler import profiler
from os import listdir
from os.path import isfile
from os.path import join as pjoin
import numpy as np
import sys

def getProfile(files):
    timeList = []
    overheadList = []
    for f in files:
        prof1 = profiler.Profiler(f[0])
        prof2 = profiler.Profiler(f[1])
        execTime = prof1.getTime('exec')
        overhead = 2 * prof2.getTime('overhead')

        timeList.append(execTime)
        overheadList.append(overhead)
    print(timeList)
    print(overheadList)
    return [np.mean(timeList), np.mean(overheadList)]

files = [pjoin(sys.argv[1], f) for f in listdir(sys.argv[1]) if 'overhead' not in f and isfile(pjoin(sys.argv[1], f))]
overhead = [pjoin(sys.argv[1], f) for f in listdir(sys.argv[1]) if 'overhead' in f and isfile(pjoin(sys.argv[1], f))]

tmp = (getProfile(zip(files, overhead)))
time = tmp[0] / 1000000.
overhead = tmp[1] / 1000000.
if int(sys.argv[2]) > 1:
    final = time + overhead
else:
    final = time


print("Average Time for {} : {} = {} + {}".format(sys.argv[1],  (final), time, overhead))


