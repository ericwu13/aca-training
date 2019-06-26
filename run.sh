NUM_MACHINE=$1
ITERATION=$2
BATCHSIZE=$3
NUM_MICRO=$4
MICROSIZE=$((BATCHSIZE/NUM_MICRO))
LOG_DIR=$5
MODEL=$6
profFile="./_result/${MODEL}_w${NUM_MACHINE}mb${MICROSIZE}.txt"
partFile="./_partition/partition_${NUM_MACHINE}_${MODEL}_${BATCHSIZE}_${NUM_MICRO}_m5000.txt"

#python profile.py -lgd ./_result -ma $NUM_MACHINE -it $ITERATION -bs $BATCHSIZE -nm $NUM_MICRO -md $MODEL

#echo "\nProfile the results to $profFile"

# 2000 -> 5000
./bin/dPartitionerCore_beta $profFile 5000 > ./_log/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}_m5000.log
./bin/dPartitionerDump $profFile 5000 $partFile


python sim.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION -md $MODEL -f $partFile > ./_log/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}_m5000.log
