"""
Simple tester for the vgg19_trainable
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import random

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from profiler import profiler
from model import model
from utils import utils
from os.path import join as pjoin

args = utils.getArgs()

logDir = args.log_dir
if not os.path.exists(logDir):
    os.makedirs(logDir)

MINI_BATCH = args.batchSize
PRINT = 0

numberMachine = args.machine
microBatchSize = MINI_BATCH // args.numMicro
numberStep = args.iters
totalTimeList = []

if args.model == 'vgg':
    mdl = model.Vgg19
elif args.model == 'resnet':
    mdl = model.ResNet
elif args.model == 'resnext':
    mdl = model.ResNeXt

print("===== Profiling Stats =====")
print("Workers Num:{0:>15}".format(numberMachine))
print("Iterations:{0:>16}".format(numberStep))
print("BatchSize:{0:>17}".format(MINI_BATCH))
print("MicroBatch Num:{0:>12}".format(args.numMicro))
print("MicroBatch Size:{0:>11}".format(microBatchSize))
print("===========================")

with tf.device('/gpu:0'):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    m = mdl(microBatchSize)
    collector = profiler.Collector(len(m.layers), numberMachine)
    print(len(m.layers))
    stages = [(i, i+1) for i in range(len(m.layers))]

    for idx, stage in enumerate(stages):
        print("\nProfiling for Stage {}({})".format(m.layerNames[idx], idx));
        m.build_single_stage(*stage)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        weightDict = {}
        for layer in tf.trainable_variables():
            name = str(layer.name.split('/')[0])
            if name not in weightDict:
                weightDict[name] = np.prod(layer.get_shape().as_list())
            else:
                weightDict[name] += np.prod(layer.get_shape().as_list())
        timeList = []
        for iter in range(numberStep+1):
            print " ------ {} / {} ...\n".format(iter+1, numberStep+1),
            fName = '/tmp/lingvo/profileData/profileResults/{}_layer_%d_%d.json'.format(args.model, idx, iter)
            inShape = m.input.shape.as_list()
            inBatch = np.random.rand(*inShape)
            feed_dict = {m.input: inBatch}
            if iter == 0:
                sessRet = sess.run(m.output, feed_dict=feed_dict)
                name = m.layerNames[idx]
                if name in weightDict:
                    weightSize = weightDict[name]
                else:
                    weightSize = 0
                activationSize = np.prod(sessRet.shape)
                collector.collectSize(activationSize, weightSize)
                if PRINT:
                    print("   ActivationShape {}".format(sessRet.shape))
                    print("   ActiationSize   {}".format(activationSize))
                    print("   WeightSize      {}".format(weightSize))
            else:
                sessRet = sess.run(m.output, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(fName, 'w') as f:
                    f.write(chrome_trace)
                    f.close()

                prof = profiler.Profiler(fName, m.layerNames[idx])
                execTime = prof.getTime('exec')
                memcpy = prof.getTime('memcpy')
                timeList.append([execTime, memcpy[0], memcpy[1]])
        timeList = np.mean(timeList, axis=0)
        totalTimeList.append(timeList)
        if PRINT:
            execTime = timeList[0]
            memIn = timeList[1]
            memOut = timeList[2]
            print("   Exec       {} us".format(execTime))
            print("   MemcpyIn   {} us".format(memIn))
            print("   MemcpyOut  {} us".format(memOut))
    assert(len(totalTimeList) == len(m.layers))
    for idx, time in enumerate(zip(totalTimeList[:-1], totalTimeList[1:])):
        # print("{} + {}".format(m.nameList[idx], m.nameList[idx+1]))
        execTime = time[0][0]
        commTime = time[0][2] + time[1][1]
        # print(commTime)
        collector.collectProfile(execTime, commTime)
    collector.collectProfile(totalTimeList[-1][0], totalTimeList[-1][2])
    collector.reset()
    collector.dump(pjoin(logDir, "{}_w{}mb{}.txt".format(args.model, numberMachine, microBatchSize)))
