DATA_DIR=./data/cifar10

TRAIN_DIR=/tmp/train_logs

python train_net_jetson_pi.py\
	--train_dir=${TRAIN_DIR} \
	--dataset_name=cifar10 \
	--dataset_split_name=train \
	--dataset_dir=${DATA_DIR} \
	--model_name=mobilenet_v2 \
        --number_of_steps=$1 \
        --batch_size=$BATCH_SIZE

