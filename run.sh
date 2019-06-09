NUM_MACHINE=$1
ITERATION=$2
BATCHSIZE=$3
NUM_MICRO=$4
MICROSIZE=$((BATCHSIZE/NUM_MICRO))
LOG_DIR=$5
MODEL=$6
profFile="./_result/${MODEL}_w${NUM_MACHINE}mb${MICROSIZE}.txt"

python profile.py -lgd ./_result -ma $NUM_MACHINE -it $ITERATION -bs $BATCHSIZE -nm $NUM_MICRO -md $MODEL

echo "\nProfile the results to $profFile"

./bin/dPartitionerCore_beta $profFile 6000
./bin/dPartitionerDump $profFile 6000 partition.txt


python sim.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION -md $MODEL

