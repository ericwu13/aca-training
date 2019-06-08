import utils
import os

args = utils.getArgs()

logDir = args.log_dir
if not os.path.exists(logDir):
    os.makedirs(logDir)
    

for n, sts in utils.parseFile(args.file).items():
    bs = args.batchSize // n
    for st in sts:
        utils.runFP(bs, st, args.iters, logDir)
        utils.runBP(bs, st, args.iters, logDir)