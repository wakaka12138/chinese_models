#!/bin/bash
module load cudnn/8.1.1.33_CUDA11.0
module load anaconda/2020.11
source activate ERNIE
python test_demo.py