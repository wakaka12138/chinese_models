#!/bin/bash
module load cudnn/7.6.5.32_CUDA10.1
module load anaconda/2020.11
source activate tensorflow
python generate.py \
    --length=200 \
    --nsamples=4 \
    --prefix='晚夏' \
    --fast_pattern \
    --save_samples \
    --save_samples_path=/data/home/scv6134/run/wangrunxin/GPT2-Chinese-old_gpt_2_chinese_before_2021_4_22/samp \
    --model_config=/data/home/scv6134/run/wangrunxin/GPT2-Chinese-old_gpt_2_chinese_before_2021_4_22/mo_del/config.json \
    --tokenizer_path=/data/home/scv6134/run/wangrunxin/GPT2-Chinese-old_gpt_2_chinese_before_2021_4_22/mo_del/vocab.txt \
    --model_path=/data/home/scv6134/run/wangrunxin/GPT2-Chinese-old_gpt_2_chinese_before_2021_4_22/mo_del
