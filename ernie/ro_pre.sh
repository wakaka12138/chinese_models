#!/bin/bash
module load cudnn/8.1.1.33_CUDA11.0
module load anaconda/2020.11
source activate ERNIE
export CUDA_VISIBLE_DEVICES=0
python predict.py --vocab_path=/data/home/scv6134/run/ernie/textcnn/robot_chat_word_dict.txt \
    --device=gpu \
    --params_path=/data/home/scv6134/run/ernie/textcnn/checkpoints/final.pdparams