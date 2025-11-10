# CIFAR-100 Long-Tail Training Scripts

This folder contains launch scripts for running the BCE Tripartite Synergistic Learning experiments.

## Available Scripts

### 1. `run_stage1.sh` - Stage 1 Training
Runs the first stage of training with BCE Tripartite Synergistic Learning.

```bash
bash scripts/run_stage1.sh [GPU_ID]
```

**Parameters:**
- `GPU_ID` (optional): GPU device ID to use (default: 0)

**Example:**
```bash
bash scripts/run_stage1.sh 0  # Run on GPU 0
```

### 2. `run_stage2.sh` - Stage 2 Training  
Runs the second stage training with MaxNorm regularization.

```bash
bash scripts/run_stage2.sh [GPU_ID] [STAGE1_PROJECT_NAME]
```

**Parameters:**
- `GPU_ID` (optional): GPU device ID to use (default: 0)
- `STAGE1_PROJECT_NAME` (optional): Name of the stage 1 project folder to load models from

**Example:**
```bash
bash scripts/run_stage2.sh 0 exp_20241109_143022  # Run stage 2 using specific stage 1 results
bash scripts/run_stage2.sh 1                      # Run stage 2 with default stage 1 models
```

### 3. `run_full_pipeline.sh` - Complete Pipeline
Runs both stages sequentially with automatic project naming.

```bash
bash scripts/run_full_pipeline.sh [GPU_ID]
```

**Parameters:**
- `GPU_ID` (optional): GPU device ID to use (default: 0)

**Example:**
```bash
bash scripts/run_full_pipeline.sh 0  # Run complete pipeline on GPU 0
```

### 4. `run_quick_test.sh` - Quick Test
Runs a quick test with reduced parameters for development and debugging.

```bash
bash scripts/run_quick_test.sh [GPU_ID]
```

**Parameters:**
- `GPU_ID` (optional): GPU device ID to use (default: 0)

**Features:**
- Reduced epochs (5 instead of 320)
- Smaller batch size (32 instead of 64)
- Lighter architecture (ResNet-18 instead of ResNet-34)
- Higher imbalance factor (0.1 instead of 0.01) for faster training

### 5. `run_experiments.sh` - Hyperparameter Experiments
Runs systematic experiments with different hyperparameter configurations.

```bash
bash scripts/run_experiments.sh [GPU_ID]
```

**Parameters:**
- `GPU_ID` (optional): GPU device ID to use (default: 0)

**Experiments:**
1. **Imbalance Factor**: Tests 0.1, 0.02, 0.01
2. **Learning Rate**: Tests 0.001, 0.01, 0.1  
3. **Architecture**: Tests ResNet-18, ResNet-34, ResNet-50

## Usage Examples

### Basic Usage
```bash
# Run stage 1 only
bash scripts/run_stage1.sh

# Run stage 2 after stage 1 completes
bash scripts/run_stage2.sh 0 exp_20241109_143022

# Run complete pipeline
bash scripts/run_full_pipeline.sh
```

### Development
```bash
# Quick test for debugging
bash scripts/run_quick_test.sh

# Run experiments
bash scripts/run_experiments.sh 0
```

### Multi-GPU Usage
```bash
# Run different experiments on different GPUs
bash scripts/run_stage1.sh 0 &  # Run on GPU 0 in background
bash scripts/run_stage1.sh 1 &  # Run on GPU 1 in background
wait  # Wait for both to complete
```

## Output Structure

Results are saved in the following structure:
```
runs/
├── main_cifar100_stage1/
│   └── [project_name]/
│       ├── log_.txt
│       ├── with_WD_model_best.paramOnly
│       ├── with_WD_model_classifier_best.paramOnly
│       └── ...
└── [stage2_project_name]/
    ├── log_.txt
    ├── MaxNorm_WD_best.paramOnly
    ├── MaxNorm_WD_classifier_best.paramOnly
    └── ...
```

## Configuration

All scripts use sensible defaults but can be customized by modifying the script parameters or directly calling the Python scripts with custom arguments.

For advanced configuration, refer to the argument parsers in `main_cifar100_stage1.py` and `main_cifar100_stage2.py`.
