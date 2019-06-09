NUM_MACHINE=$1
ITERATION=$2
BATCHSIZE=$3
NUM_MICRO=$4
MICROSIZE=$((BATCHSIZE/NUM_MICRO))
LOG_DIR=$5
profFile="./_result/vgg_w${NUM_MACHINE}mb${MICROSIZE}.txt"

echo "Profile the results to $profFile"
python profile.py -lgd ./_result -ma $NUM_MACHINE -it $ITERATION -bs $BATCHSIZE -nm $NUM_MICRO

./bin/dPartitionerCore_beta $profFile 6000
./bin/dPartitionerDump $profFile 6000 partition.txt


python sim.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION

