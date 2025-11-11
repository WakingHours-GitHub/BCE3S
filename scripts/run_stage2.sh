#!/bin/bash
"""
Launch script for CIFAR-100 Long-Tail Training Stage 2
Usage: bash scripts/run_stage2.sh [GPU_ID] [STAGE1_PROJECT_NAME]
"""

# Default configuration
GPU_ID=${1:-0}
STAGE1_PROJECT=${2:-""}
# PROJECT_NAME="stage2_$(date +%Y%m%d_%H%M%S)"
PROJECT_NAME="main_cifar100_stage2"

echo "Starting CIFAR-100 Stage 2 Training..."
echo "GPU: $GPU_ID"
echo "Stage 1 Project: $STAGE1_PROJECT"
echo "Project: $PROJECT_NAME"

# Build stage1_root path
if [ -n "$STAGE1_PROJECT" ]; then
    STAGE1_ROOT="runs/main_cifar100_stage1/$STAGE1_PROJECT"
else
    STAGE1_ROOT="runs/main_cifar100_stage1"
fi

python main_cifar100_stage2.py \
    --gpu $GPU_ID \
    --project_name $PROJECT_NAME \
    --stage1_root $STAGE1_ROOT \
    --batch_size 64 \
    --epochs 20 \
    --lr 0.0001 \
    --weight_decay 0 \
    --imb_factor 0.01 \
    --encoder_layers 34 \
    --embedding_dim 512 \
    --model_type last \
    --maxnorm_thresh 0.1 \
    --cb_beta 0.9999 \
    --cb_gamma 2.0 \
    --cb_r 0.4 \
    --seed 0

echo "Stage 2 training completed!"
echo "Results saved in: runs/$PROJECT_NAME"
