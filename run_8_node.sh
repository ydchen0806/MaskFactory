#!/bin/bash

# 总共的GPU数量
NUM_GPUS=8

for i in $(seq 0 $(($NUM_GPUS - 1)))
do
    echo "Starting task on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python3 /data/ydchen/VLP/MasaCtrl/parrellel_diffusion.py  --local_rank=$i --out_dir_base=/h3cstore_ns/ydchen/mask_edit/zero123_0504 --filter_value=3 &
done

wait
echo "All tasks completed."
