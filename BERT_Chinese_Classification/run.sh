#!/bin/bash
module load cudnn/7.6.3_CUDA10.0
module load anaconda/2020.11
source activate tf1.0
python run_classifier.py \
--data_dir=/data/home/scv6134/run/BERT_Chinese_Classification/data \
--task_name=sim \
--vocab_file=/data/home/scv6134/run/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/data/home/scv6134/run/chinese_L-12_H-768_A-12/bert_config.json \
--output_dir=/data/home/scv6134/run/BERT_Chinese_Classification/sim_model \
--do_train=true \
--do_eval=true \
--init_checkpoint=/data/home/scv6134/run/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=300 \
--train_batch_size=8 \
--learning_rate=5e-5 \
--num_train_epochs=1.0