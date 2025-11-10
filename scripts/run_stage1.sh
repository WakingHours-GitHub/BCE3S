#!/bin/bash
"""
Launch script for CIFAR-100 Long-Tail Training Stage 1
Usage: bash scripts/run_stage1.sh [GPU_ID]
"""

# Default configuration
GPU_ID=${1:-0}
PROJECT_NAME="exp_$(date +%Y%m%d_%H%M%S)"

echo "Starting CIFAR-100 Stage 1 Training..."
echo "GPU: $GPU_ID"
echo "Project: $PROJECT_NAME"

python main_cifar100_stage1.py \
    --gpu $GPU_ID \
    --project_name $PROJECT_NAME \
    --batch_size 64 \
    --epochs 320 \
    --lr 0.01 \
    --weight_decay 5e-3 \
    --imb_factor 0.01 \
    --encoder_layers 34 \
    --embedding_dim 512 \
    --project_dim 128 \
    --print_freq 10 \
    --seed 0

echo "Stage 1 training completed!"
echo "Results saved in: runs/main_cifar100_stage1/$PROJECT_NAME"
