#!/bin/bash
module load cudnn/8.1.1.33_CUDA11.0
module load anaconda/2020.11
source activate ERNIE
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --vocab_path=/data/home/scv6134/run/ernie/textcnn/robot_chat_word_dict.txt \
    --init_from_ckpt=/data/home/scv6134/run/ernie/textcnn/textcnn.pdparams \
    --device=gpu \
    --lr=5e-5 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir=/data/home/scv6134/run/ernie/textcnn/checkpoints \
    --data_path=/data/home/scv6134/run/ernie/textcnn/RobotChat
