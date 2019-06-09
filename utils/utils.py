import numpy as np

from argparse import ArgumentParser
from os.path import join as pjoin



def getArgs():
    parser = ArgumentParser()
    parser.add_argument('-lgd', '--log_dir', type=str, default='/home/ACA/timeline/')
    parser.add_argument('-bs', '--batchSize', type=int, default=64)
    parser.add_argument('-nm', '--numMicro', type=int, default=4)
    parser.add_argument('-it', '--iters', type=int, default=10)
    parser.add_argument('-f', '--file', type=str, default='partition.txt')
    parser.add_argument('-md', '--model', type=str, default='vgg')
    parser.add_argument('-ma', '--machine', type=int, default=4)
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


