#!/bin/bash

python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12345 generate_DB.py \
    --tag debug \
    --cfg configs/swin_tiny_patch4_window7_224.yaml \
    --batch-size 1 \
    --resume checkpoints/swin_tiny_patch4_window7_224.pth \
    --data-path database/data
