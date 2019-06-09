import os

from utils import utils
from simulator import simulator as sim
from model import vgg19_bp as vgg19

args = utils.getArgs()

logDir = args.log_dir
if not os.path.exists(logDir):
    os.makedirs(logDir)

print("========== Stats ==========")
for arg in vars(args):
        print arg, getattr(args, arg)
print("===========================")


microBatchSize = args.batchSize // args.numMicro
machineNum, machineDict = utils.parseFile(args.file)

numStages = 0
for n, nts in machineDict.items():
    numStages += len(nts)
print("Number of Stages: {}".format(numStages))

if args.model == 'vgg':
    model = vgg19.Vgg19
elif args.model == 'resnet':
    pass

s = sim.Simulator(model, args.numMicro, numStages, args.iters, logDir, machineNum)

for n, sts in machineDict.items():
    bs = microBatchSize // n
    if bs == 0:
        bs = 1
    for st in sts:
        print(st[1])
        print("\nProfiling for Stage ({}, {}) with bs({})".format(st[1][0], st[1][1], bs))
        s.runFP(bs, st)
        s.runBP(bs, st)

s.runProfile()
