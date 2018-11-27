python viz.py saved_model.pb log

python run.py tensorflow_inception_graph.pb

## Running slim networks with profiling

DATA_DIR=/tmp/data/cifar10
python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="${DATA_DIR}"

TRAIN_DIR=/tmp/train_logs

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATA_DIR} \
    --model_name=mobilenet_v2