"""
Simple tester for the vgg19_trainable
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import random
import vgg19_bp as vgg19
import os


batchSize = 128
stages = [(0,3), (3,6), (6,11), (11,27)]

logDir = '/home/ACA/_timeline/vgg_{}_stages/'.format(str(len(stages)))
if not os.path.exists(logDir):
    os.makedirs(logDir)

with tf.device('/gpu:0'):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    vgg = vgg19.Vgg19(batchSize)
    
    for idx, stage in enumerate(stages):
        vgg.build_single_stage(*stage)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        ops = vgg.single_stage_bp()
        
        for iter in range(11):
            inShape = vgg.input.shape.as_list()
            inBatch = np.random.rand(*inShape)
            
            if idx == len(stages)-1:
                outBatch = [[1 if i==int(random.random()*1000) else 0 for i in range(1000)] for n in range(batchSize)]
            else:
                outShape = vgg.output.shape.as_list()
                outBatch = np.random.rand(*outShape)
                
            feed_dict = {vgg.input: inBatch, vgg._output: outBatch}
            sessRet = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
            
            if iter != 0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(logDir+'bp_stage_{}_{}.json'.format(idx, iter), 'w') as f:
                    f.write(chrome_trace)
