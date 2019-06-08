import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.python.client import timeline
from model import vgg19_bp as vgg19
from profiler import profiler
from argparse import ArgumentParser
from os import listdir
from os.path import isfile
from os.path import join as pjoin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getArgs():
    parser = ArgumentParser()
    parser.add_argument('-lgd', '--log_dir', type=str, default='/home/ACA/timeline/')
    parser.add_argument('-bs', '--batchSize', type=int, default=64)
    parser.add_argument('-nm', '--numMicro', type=int, default=4)
    parser.add_argument('-it', '--iters', type=int, default=10)
    parser.add_argument('-f', '--file', type=str, default='partition.txt')
    args = parser.parse_args()

    return args

def parseFile(fileName):
    stages = []
    machineNum  = []
    idx = 0
    with open(fileName) as fp:
        for line in fp:
            st = tuple([int(i) for i in line.strip('\n').split(' ')])
            if len(stages) > 0 and st == stages[-1][1]:
                stages[-1][0] += 1
                machineNum[idx-1] += 1
            else:
                idx = idx + 1
                machineNum.append(1)
                stages.append([1, st])

    ret = dict() # n machines: list of (stage i, (layer a, layer b))
    for i, j in enumerate(stages):
        if j[0] in ret.keys():
            ret[j[0]].append((i, j[1]))
        else:
            ret[j[0]] = [(i, j[1])]

    return machineNum, ret

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

class Simulator:
    def __init__(self, numMicro, numStages, iters, logDir, machineNum):
        self.numMicro = numMicro
        self.numStages = numStages
        self.iters = iters
        self.logDir = logDir
        self.machineNum = machineNum

    def runBP(self, batchSize, stage):
        with tf.device('/gpu:0'):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            vgg = vgg19.Vgg19(batchSize)
            vgg.build_single_stage(*stage[1])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            ops = vgg.single_stage_bp()

            for iter in range(self.iters+1):
                print " ------ {} / {} ...\n".format(iter+1, self.iters+1),
                inShape = vgg.input.shape.as_list()
                inBatch = np.random.rand(*inShape)

                if stage[1] == len(vgg.layers)-1:
                    outBatch = [[1 if i==int(random.random()*1000) else 0 for i in range(1000)] for n in range(batchSize)]
                else:
                    outShape = vgg.output.shape.as_list()
                    outBatch = np.random.rand(*outShape)

                feed_dict = {vgg.input: inBatch, vgg._output: outBatch}
                sessRet = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

                if iter != 0:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                    with open(pjoin(self.logDir,'bp_stage_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                        f.write(chrome_trace)
                        f.close()

    def runFP(self, batchSize, stage):
        # print(stage)
        with tf.device('/gpu:0'):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            vgg = vgg19.Vgg19(batchSize)
            vgg.build_single_stage(*stage[1])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            for iter in range(self.iters+1):
                print " ------ {} / {} ...\n".format(iter+1, self.iters+1),
                inShape = vgg.input.shape.as_list()
                inBatch = np.random.rand(*inShape)

                feed_dict = {vgg.input: inBatch}
                sessRet = sess.run(vgg.output, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

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


        fpMeanList = getProfile('fp_stage', self.numStages, self.iters, self.logDir)
        bpMeanList = getProfile('bp_stage', self.numStages, self.iters, self.logDir)
        print("\nMean FP TimeList")
        print(fpMeanList)
        print("\nMean BP TimeList")
        print(bpMeanList)

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

        print("\nTotal Time: {} s".format((fp+bp)/10**6.))

if __name__ == "__main__":
    sim = Simulator(16, 4, 10, '_timeline/vgg')
    sim.runProfile()


