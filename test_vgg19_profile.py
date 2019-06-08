"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import sys
import numpy as np
import random
import os

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from profiler import profiler
from model import vgg19_bp as vgg19

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MINI_BATCH = 256
PRINT = 0

numberMachine = sys.argv[1]
numberMicro = MINI_BATCH // (int)(sys.argv[2])
numberStep = int(sys.argv[3])
totalTimeList = []


with tf.device('/gpu:0'):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    vgg = vgg19.Vgg19(numberMicro)
    collector = profiler.Collector(len(vgg.layers), numberMachine)
    print(len(vgg.layers))
    stages = [(i, i+1) for i in range(len(vgg.layers))]

    for idx, stage in enumerate(stages):
        print("\nProfiling for Stage {}({})".format(vgg.layerNames[idx], idx));
        vgg.build_single_stage(*stage)
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
        for iter in range(numberStep):
            print " ------ {} / {} ...\n".format(iter+1, numberStep),
            fName = '/tmp/lingvo/profileData/profileResults/vgg_layer_%d_%d.json'.format(idx, iter)
            inShape = vgg.input.shape.as_list()
            inBatch = np.random.rand(*inShape)
            feed_dict = {vgg.input: inBatch}
            if iter == 0:
                sessRet = sess.run(vgg.output, feed_dict=feed_dict)
                name = vgg.layerNames[idx]
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
                sessRet = sess.run(vgg.output, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(fName, 'w') as f:
                    f.write(chrome_trace)
                    f.close()

                prof = profiler.Profiler(fName, vgg.layerNames[idx])
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
    assert(len(totalTimeList) == len(vgg.layers))
    for idx, time in enumerate(zip(totalTimeList[:-1], totalTimeList[1:])):
        # print("{} + {}".format(vgg.nameList[idx], vgg.nameList[idx+1]))
        execTime = time[0][0]
        commTime = time[0][2] + time[1][1]
        # print(commTime)
        collector.collectProfile(execTime, commTime)
    collector.collectProfile(totalTimeList[-1][0], totalTimeList[-1][2])
    collector.reset()
    collector.dump("result/vgg_w{}mb{}.txt".format(numberMachine, numberMicro))
