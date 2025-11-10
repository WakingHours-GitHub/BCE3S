#!/bin/bash
"""
Quick test script for development and debugging
Usage: bash scripts/run_quick_test.sh [GPU_ID]
"""

GPU_ID=${1:-0}
PROJECT_NAME="quick_test_$(date +%Y%m%d_%H%M%S)"

echo "Running Quick Test (Stage 1 only, reduced epochs)..."
echo "GPU: $GPU_ID"
echo "Project: $PROJECT_NAME"

python main_cifar100_stage1.py \
    --gpu $GPU_ID \
    --project_name $PROJECT_NAME \
    --batch_size 32 \
    --epochs 5 \
    --lr 0.01 \
    --weight_decay 5e-3 \
    --imb_factor 0.1 \
    --encoder_layers 18 \
    --embedding_dim 256 \
    --project_dim 64 \
    --print_freq 1 \
    --seed 0

echo "Quick test completed!"
echo "Results saved in: runs/main_cifar100_stage1/$PROJECT_NAME"
