import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
import tensorflow as tf

from tensorflow.python.client import timeline
from profiler import profiler
from os import listdir
from os.path import isfile
from os.path import join as pjoin

def getProfile(name, numStages, iters, logDir):
    stagesMeanList = []
    for m in range(numStages):
        timeList = []
        for i in range(iters):
            f = pjoin(logDir, '{}_{}_{}.json'.format(name, m, i+1))
            prof = profiler.Profiler(f)
            execTime = prof.getTime('exec')
            memcpy = prof.getTime('memcpy')
            timeList.append([execTime, memcpy[0], memcpy[1]])
        stagesMeanList.append(list(np.mean(timeList, axis=0, dtype=int)))
    assert(len(stagesMeanList) == numStages)
    assert(len(stagesMeanList[0]) == 3)
    return stagesMeanList

def getOverHead(numStages, iters, logDir):
    stagesMeanList = []
    for m in range(numStages):
        timeList = []
        for i in range(iters):
            f = pjoin(logDir, 'dp_overhead_{}_{}.json'.format(m, i+1))
            prof = profiler.Profiler(f)
            execTime = 2 * prof.getTime('overhead')
            print(execTime/2.)
            timeList.append(execTime)
        stagesMeanList.append(np.mean(timeList, dtype=int))
    print(stagesMeanList)
    return stagesMeanList

class Simulator:
    def __init__(self, model, numMicro, numStages, iters, logDir, machineNum):
        self.model = model
        self.numMicro = numMicro
        self.numStages = numStages
        self.iters = iters
        self.logDir = logDir
        self.machineNum = machineNum

    def runBP(self, batchSize, stage):
        with tf.device('/gpu:0'):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            m = self.model(batchSize)
            m.build_single_stage(*stage[1])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            ops = m.single_stage_bp()

            for iter in range(self.iters+1):
                print " ------ {} / {} ...\n".format(iter+1, self.iters+1),
                inShape = m.input.shape.as_list()
                inBatch = np.random.rand(*inShape)

                if stage[1] == len(m.layers)-1:
                    outBatch = [[1 if i==int(random.random()*1000) else 0 for i in range(1000)] for n in range(batchSize)]
                else:
                    outShape = m.output.shape.as_list()
                    outBatch = np.random.rand(*outShape)

                feed_dict = {m.input: inBatch, m._output: outBatch}
                sessRet = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

                if iter != 0:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                    with open(pjoin(self.logDir,'bp_stage_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                        f.write(chrome_trace)
                        f.close()
                    v = sess.run(tf.trainable_variables(), options=options, run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                    with open(pjoin(self.logDir,'dp_overhead_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                        f.write(chrome_trace)
                        f.close()

    def runFP(self, batchSize, stage):
        # print(stage)
        with tf.device('/gpu:0'):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            m= self.model(batchSize)
            m.build_single_stage(*stage[1])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            for iter in range(self.iters+1):
                print " ------ {} / {} ...\n".format(iter+1, self.iters+1),
                inShape = m.input.shape.as_list()
                inBatch = np.random.rand(*inShape)

                feed_dict = {m.input: inBatch}
                sessRet = sess.run(m.output, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

                if iter != 0:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                    with open(pjoin(self.logDir,'fp_stage_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                        f.write(chrome_trace)
                        f.close()

    def runProfile(self):
        # calculate fp and bp execution, communication time
        fpFiles = [f for f in listdir(self.logDir) if 'fp_stage' in f and isfile(pjoin(self.logDir, f))]
        bpFiles = [f for f in listdir(self.logDir) if 'bp_stage' in f and isfile(pjoin(self.logDir, f))]
        ohFiles = [f for f in listdir(self.logDir) if 'dp_overhead' in f and isfile(pjoin(self.logDir, f))]


        fpMeanList = getProfile('fp_stage', self.numStages, self.iters, self.logDir)
        bpMeanList = getProfile('bp_stage', self.numStages, self.iters, self.logDir)
        ohMeanList = getOverHead(self.numStages, self.iters, self.logDir)
        print("\nMean FP TimeList")
        print(fpMeanList)
        print("\nMean BP TimeList")
        print(bpMeanList)
        print("\nMean DPOH TimeList")
        print(ohMeanList)

        fT  = []
        for idx, time in enumerate(zip(fpMeanList[:-1], fpMeanList[1:])):
            fT.append([time[0][0], time[0][2] * self.machineNum[idx] + time[1][1] * self.machineNum[idx+1]])
        fT.append([fpMeanList[-1][0], bpMeanList[-1][1]])

        bT  = []
        for idx, time in enumerate(zip(reversed(bpMeanList[1:]), reversed(bpMeanList[:-1]))):
            bT.append([time[0][0], time[0][2] * self.machineNum[::-1][idx] + time[1][1]] * self.machineNum[::-1][idx+1])
        bT.append([bpMeanList[0][0], bpMeanList[0][2]])

        print("\nFP TimeList")
        print(fT)
        print("\nBP TimeList")
        print(bT)

        timeSteps = 2 * (self.numStages - 1) + self.numMicro
        timeChart = [[] for i in range(timeSteps)]

        for jdx, t in enumerate(fT):
            for idx in range(jdx*2, timeSteps):
                if idx < jdx*2 + self.numMicro:
                    # print(timeChart[idx][jdx])
                    timeChart[idx].append(t[0])
                if idx > jdx*2 and idx < jdx*2 + self.numMicro:
                    timeChart[idx].append(t[1])
        finalFP = [max(t) for t in timeChart]
        fp = np.sum(finalFP)
        print "\nTime Chart for FP"
        for t in timeChart:
            print(t)
        print "\nFP for each step: ",
        print(finalFP)

        timeChart = [[] for i in range(timeSteps)]
        timeChart[0].append(fT[-1][1])
        for jdx, t in enumerate(bT):
            for idx in range(jdx*2, timeSteps):
                if idx < jdx*2 + self.numMicro:
                    # print(timeChart[idx][jdx])
                    timeChart[idx].append(t[0])
                if idx > jdx*2 and idx < jdx*2 + self.numMicro:
                    timeChart[idx].append(t[1])
        finalBP = [max(t) for t in timeChart]
        bp = np.sum(finalBP)
        print "\nTime Chart for BP"
        for t in timeChart:
            print(t)
        print "\nBP for each step: ",
        print(finalBP)

        finalTime = fp + bp
        print("Time without overhead: {} s".format(finalTime / 10**6.))

        for idx, dp in enumerate(self.machineNum):
            if( dp > 1 ):
                finalTime += ohMeanList[idx]

        print("\nTotal Time: {} s".format(finalTime/10**6.))

if __name__ == "__main__":
    sim = Simulator(16, 4, 10, '_timeline/vgg')
    sim.runProfile()
