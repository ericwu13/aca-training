from argparse import ArgumentParser
from os.path import join as pjoin

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import random
import vgg19_bp as vgg19


def getArgs():
    parser = ArgumentParser()
    parser.add_argument('-lgd', '--log_dir', type=str, default='/home/ACA/timeline/')
    parser.add_argument('-bs', '--batchSize', type=int, default=64)
    parser.add_argument('-it', '--iters', type=int, default=10)
    parser.add_argument('-f', '--file', type=str, default='partition.txt')
    args = parser.parse_args()
    
    return args
    
def parseFile(fileName):
    stages = []
    with open(fileName) as fp:
        for line in fp:
            st = tuple([int(i) for i in line.strip('\n').split(' ')])
            if len(stages) > 0 and st == stages[-1][1]:
                stages[-1][0] += 1
            else:
                stages.append([1, st])
    
    ret = dict() # n machines: list of (stage i, (layer a, layer b))
    for i, j in enumerate(stages):
        if j[0] in ret.keys():
            ret[j[0]].append((i, j[1]))
        else:
            ret[j[0]] = [(i, j[1])]
    
    return ret

def runBP(batchSize, stage, iters, logDir):
    with tf.device('/gpu:0'):
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        vgg = vgg19.Vgg19(batchSize)
        vgg.build_single_stage(*stage[1])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        ops = vgg.single_stage_bp()
        
        for iter in range(iters+1):
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
                with open(pjoin(logDir,'bp_stage_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                    f.write(chrome_trace)
        
def runFP(batchSize, stage, iters, logDir):
    print(stage)
    with tf.device('/gpu:0'):
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        vgg = vgg19.Vgg19(batchSize)
        vgg.build_single_stage(*stage[1])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for iter in range(iters+1):
            inShape = vgg.input.shape.as_list()
            inBatch = np.random.rand(*inShape)
                
            feed_dict = {vgg.input: inBatch}
            sessRet = sess.run(vgg.output, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
            
            if iter != 0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(pjoin(logDir,'fp_stage_{}_{}.json'.format(stage[0], iter)), 'w') as f:
                    f.write(chrome_trace)