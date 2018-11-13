DATA_DIR=./data/flowers

TRAIN_DIR=/tmp/train_logs

python train_net_jetson_pi.py\
	--train_dir=${TRAIN_DIR} \
	--dataset_name=flowers \
	--dataset_split_name=train \
	--dataset_dir=${DATA_DIR} \
	--model_name=inception_v3 \
        --number_of_steps=$1 \
        --batch_size=$BATCH_SIZE

