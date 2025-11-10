#!/bin/bash
"""
Complete pipeline script for CIFAR-100 Long-Tail Training (Stage 1 + Stage 2)
Usage: bash scripts/run_full_pipeline.sh [GPU_ID]
"""

# Default configuration
GPU_ID=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGE1_PROJECT="stage1_$TIMESTAMP"
STAGE2_PROJECT="stage2_$TIMESTAMP"

echo "========================================="
echo "Starting CIFAR-100 Full Training Pipeline"
echo "========================================="
echo "GPU: $GPU_ID"
echo "Timestamp: $TIMESTAMP"
echo "Stage 1 Project: $STAGE1_PROJECT"
echo "Stage 2 Project: $STAGE2_PROJECT"
echo ""

# Stage 1 Training
echo "Starting Stage 1 Training..."
python main_cifar100_stage1.py \
    --gpu $GPU_ID \
    --project_name $STAGE1_PROJECT \
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

if [ $? -eq 0 ]; then
    echo "Stage 1 completed successfully!"
    echo ""
else
    echo "Stage 1 failed! Exiting..."
    exit 1
fi

# Stage 2 Training
echo "Starting Stage 2 Training..."
python main_cifar100_stage2.py \
    --gpu $GPU_ID \
    --project_name $STAGE2_PROJECT \
    --stage1_root "runs/main_cifar100_stage1/$STAGE1_PROJECT" \
    --batch_size 64 \
    --epochs 40 \
    --lr 0.0001 \
    --weight_decay 0 \
    --imb_factor 0.01 \
    --encoder_layers 34 \
    --embedding_dim 512 \
    --model_type best \
    --maxnorm_thresh 0.1 \
    --cb_beta 0.99999 \
    --cb_gamma 2.0 \
    --cb_r 0.4 \
    --seed 0

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Full Pipeline Completed Successfully!"
    echo "========================================="
    echo "Stage 1 Results: runs/main_cifar100_stage1/$STAGE1_PROJECT"
    echo "Stage 2 Results: runs/$STAGE2_PROJECT"
else
    echo "Stage 2 failed!"
    exit 1
fi
