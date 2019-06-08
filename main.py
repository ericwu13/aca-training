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
print(numStages)
print(machineNum)
sim = utils.Simulator(args.numMicro, numStages, args.iters, logDir, machineNum)

for n, sts in machineDict.items():
    bs = microBatchSize // n
    for st in sts:
        sim.runFP(bs, st)
        sim.runBP(bs, st)

sim.runProfile()
