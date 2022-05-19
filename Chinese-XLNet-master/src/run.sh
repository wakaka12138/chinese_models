#!/bin/bash
module load cudnn/7.6.3_CUDA10.0
module load anaconda/2020.11
source activate tf1.0
XLNET_DIR=/data/home/scv6134/run/wangrunxin/Chinese-XLNet-master
MODEL_DIR=/data/home/scv6134/run/wangrunxin/Chinese-XLNet-master/models
DATA_DIR=/data/home/scv6134/run/wangrunxin/Chinese-XLNet-master/data
RAW_DIR=/data/home/scv6134/run/wangrunxin/ernie
TPU_NAME=v2-xlnet
TPU_ZONE=us-central1-b
python -u run_classifier.py \
	--spiece_model_file=/data/home/scv6134/run/wangrunxin/Chinese-XLNet-master/spiece.model \
	--model_config_path=${XLNET_DIR}/xlnet_config.json \
	--init_checkpoint=${XLNET_DIR}/xlnet_model.ckpt \
	--task_name=csc \
	--do_train=True \
	--do_eval=True \
	--eval_all_ckpt=False \
	--uncased=False \
	--data_dir=${RAW_DIR} \
	--output_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--train_batch_size=48 \
	--eval_batch_size=48 \
	--num_hosts=1 \
	--num_core_per_host=1 \
	--num_train_epochs=3 \
	--max_seq_length=256 \
	--learning_rate=2e-5 \
	--save_steps=5000 \
	--use_tpu=False \
	--num_core_per_host=1 \
	--tpu_zone=${TPU_ZONE}