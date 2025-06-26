#!/bin/bash
export PYTORCH_ENABLE_FUNC_IMPL=1 && \
export PYTORCH_DDP_NO_REBUILD_BUCKETS=1 && \
export HF_HOME=/cpfs/user/weijian/cache && \
export TORCH_NCCL_IB_TIMEOUT=23 && \
export NCCL_TIMEOUT=3600 && \
export SETUPTOOLS_USE_DISTUTILS=local && \

torchrun --standalone --nproc_per_node=8 train_mf.py \
    --detach_tgt=1 \
    --outdir=./image_experiments/mf/MF03 \
    --data=./data/cifar10-32x32.zip \
    --cond=0 --arch=ddpmpp --lr 10e-4 --batch 8
