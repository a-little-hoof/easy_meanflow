#!/bin/bash    
torchrun --standalone --nproc_per_node=8 train_mf.py \
    --detach_tgt=1 \
    --outdir=./image_experiments/mf/MF03 \
    --data=./data/cifar10-32x32.zip \
    --cond=0 --arch=ddpmpp --lr 10e-4 --batch 8
