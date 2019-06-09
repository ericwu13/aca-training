NUM_MACHINE=$1
ITERATION=$2
BATCHSIZE=$3
NUM_MICRO=$4
MICROSIZE=$((BATCHSIZE/NUM_MICRO))
LOG_DIR=$5
profFile="./_result/vgg_w${NUM_MACHINE}mb${MICROSIZE}.txt"

echo "Profile the results to $profFile"
python test_vgg19_profile.py $NUM_MACHINE $ITERATION $BATCHSIZE $NUM_MICRO

./dPartitionerCore_beta $profFile 6000
./dPartitionerDump $profFile 6000 partition.txt

cat partition.txt

python main.py -lgd $LOG_DIR -bs $BATCHSIZE -nm $NUM_MICRO -it $ITERATION

