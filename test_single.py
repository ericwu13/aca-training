import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import random
import sys

from tensorflow.python.client import timeline
from model import model
from utils import utils
from os.path import join as pjoin

args = utils.getArgs()

logDir = args.log_dir
if not os.path.exists(logDir):
    os.makedirs(logDir)

if args.model == 'vgg':
    mdl = model.Vgg19
elif args.model == 'resnet':
    mdl = model.ResNet
elif args.model == 'resnext':
    mdl = model.ResNeXt

batchSize = args.batchSize


with tf.device('/gpu:0'):
    print("Profiling single GPU on {} with bs({}) ...".format(args.model, batchSize))

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    m = mdl(batchSize)
    m.build_single_stage(0, len(m.layers))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ops = m.single_stage_bp()

    for iter in range(args.iters+1):
        print " ------ {} / {} ...\n".format(iter+1, args.iters+1),
        inShape = m.input.shape.as_list()
        inBatch = np.random.rand(*inShape)

        outBatch = [[1 if i==int(random.random()*1000) else 0 for i in range(1000)] for n in range(batchSize)]

        feed_dict = {m.input: inBatch, m._output: outBatch}
        sessRet = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

        if iter != 0:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
            with open(pjoin(logDir,'single_gpu_{}.json'.format(iter)), 'w') as f:
                f.write(chrome_trace)
                f.close()
            v = sess.run(tf.trainable_variables(), options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
            with open(pjoin(logDir,'single_gpu_overhead_{}.json'.format(iter)), 'w') as f:
                f.write(chrome_trace)
                f.close()
