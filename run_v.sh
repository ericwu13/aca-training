NUM_MACHINE=$1
ITERATION=$2
BATCHSIZE=$3
NUM_MICRO=$4
MICROSIZE=$((BATCHSIZE/NUM_MICRO))
LOG_DIR=$5
MODEL=$6
profFile="./_result/${MODEL}_w${NUM_MACHINE}mb${MICROSIZE}.txt"
partFile="./_parth/partition_${NUM_MACHINE}_${MODEL}_${BATCHSIZE}_${NUM_MICRO}"

#python profile.py -lgd ./_result -ma $NUM_MACHINE -it $ITERATION -bs $BATCHSIZE -nm $NUM_MICRO -md $MODEL

#echo "\nProfile the results to $profFile"

./bin/dPartitionerCore_beta $profFile 3000 > ./_log/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}.log
./bin/dPartitionerDump $profFile 3000 "$partFile_3000.txt"
python sim.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION -md $MODEL -f "$partFile_3000.txt" > ./_logh/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}_m3000.log


./bin/dPartitionerCore_beta $profFile 6000 > ./_log/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}.log
./bin/dPartitionerDump $profFile 6000 "$partFile_6000.txt"
python sim.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION -md $MODEL -f "$partFile_6000.txt" > ./_logh/${MODEL}_${NUM_MACHINE}_${BATCHSIZE}_${NUM_MICRO}_m6000.log
