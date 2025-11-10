#!/bin/bash
"""
Hyperparameter sweep script for experimenting with different configurations
Usage: bash scripts/run_experiments.sh [GPU_ID]
"""

GPU_ID=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting Hyperparameter Experiments..."
echo "GPU: $GPU_ID"
echo "Timestamp: $TIMESTAMP"

# Experiment 1: Different imbalance factors
echo "Experiment 1: Testing different imbalance factors..."
for imb_factor in 0.1 0.02 0.01; do
    project_name="exp1_imb${imb_factor}_$TIMESTAMP"
    echo "Running with imbalance factor: $imb_factor"
    
    python main_cifar100_stage1.py \
        --gpu $GPU_ID \
        --project_name $project_name \
        --imb_factor $imb_factor \
        --epochs 100 \
        --batch_size 64 \
        --lr 0.01 \
        --seed 0
done

# Experiment 2: Different learning rates
echo "Experiment 2: Testing different learning rates..."
for lr in 0.001 0.01 0.1; do
    project_name="exp2_lr${lr}_$TIMESTAMP"
    echo "Running with learning rate: $lr"
    
    python main_cifar100_stage1.py \
        --gpu $GPU_ID \
        --project_name $project_name \
        --lr $lr \
        --epochs 100 \
        --batch_size 64 \
        --imb_factor 0.01 \
        --seed 0
done

# Experiment 3: Different architectures
echo "Experiment 3: Testing different ResNet architectures..."
for layers in 18 34 50; do
    project_name="exp3_resnet${layers}_$TIMESTAMP"
    echo "Running with ResNet-$layers"
    
    python main_cifar100_stage1.py \
        --gpu $GPU_ID \
        --project_name $project_name \
        --encoder_layers $layers \
        --epochs 100 \
        --batch_size 64 \
        --lr 0.01 \
        --imb_factor 0.01 \
        --seed 0
done

echo "All experiments completed!"
echo "Results saved with timestamp: $TIMESTAMP"
