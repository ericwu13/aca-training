from utils import utils
import os

args = utils.getArgs()

logDir = args.log_dir
if not os.path.exists(logDir):
    os.makedirs(logDir)

microBatchSize = args.batchSize // args.numMicro
machineNum, machineDict = utils.parseFile(args.file)
numStages = 0
for n, nts in machineDict.items():
    numStages += len(nts)
print("Number of Stages: {}".format(numStages))
sim = utils.Simulator(args.numMicro, numStages, args.iters, logDir, machineNum)

for n, sts in machineDict.items():
    bs = microBatchSize // n
    if bs == 0:
        bs = 1
    for st in sts:
        print(st[1])
        print("\nProfiling for Stage ({}, {}) with bs({})".format(st[1][0], st[1][1], bs))
        sim.runFP(bs, st)
        sim.runBP(bs, st)

sim.runProfile()
