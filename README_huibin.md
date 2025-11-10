# ![Logo image](https://github.com/user-attachments/assets/26a93348-5bed-4107-9113-0dfe6ee707e8) \[ICML 2025\] BCE vs. CE in Deep Feature Learning

This is the **main code** for the paper "[BCE vs. CE in Deep Feature Learning](https://icml.cc/virtual/2025/poster/44778)".

Forty-Second International Conference on Machine Learning \(ICML\), 2025.

🎬 None | 💻 None | 🔥 [Poster](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/BCE%20vs.%20CE.PNG).

# Overview
* We provide the first theoretical proof that BCE can also lead to the NC, i.e., maximizing the compactness and distinctiveness.
* We find that BCE performs better than CE in enhancement of intra-class compactness and inter-class distinctiveness across sample features, and, BCE can explicitly enhance the feature properties, while CE only implicitly enhance them.
* We point out that when training models with BCE, the classifier biases play substantial role in enhancing the feature properties, while in the training with CE, it almost does not work.
* We conduct extensive experiments, and find that, compared to CE, BCE can more quickly lead to NC on the training dataset and achieves better feature compactness and distinctiveness, resulting in higher classification performance on the test dataset.

# Getting Started
## Requirements
* CUDA == 12.4
* python == 3.8.20
* torch == 2.4.0
* torchvision == 0.19.0
* scipy == 1.10.1
* numpy == 1.24.3

## Preparing Datasets
By default, the main code assumes the datasets for MNIST, CIFAR10 and CIFAR100 are stored under `./data/`. If the datasets are not there, they will be automatically downloaded from `torchvision.datasets`. User may change this default location of datasets in `args.py` through the argument `--data_dir`.

The directory should be look like: 
```
./data
├── cifar-10-batches-py
├── cifar-100-python
├── MNIST
    └── raw
```

# Training and Testing
## Training with SGD
```
$ python train.py --gpu_id 0 --model <ResNet18 or DenseNet121 or ViT> --dataset <mnist or cifar10 or cifar100> --optimizer SGD --loss <CE or BCE> --batch_size 128 --lr 0.01
```

## Training with AdamW
```
$ python train.py --gpu_id 0 --model <ResNet18 or DenseNet121 or ViT> --dataset <mnist or cifar10 or cifar100> --optimizer AdamW --loss <CE or BCE> --batch_size 128 --lr 0.01
```

**Note**: For each epoch during training, the model checkpoints will be saved in the directory `./model_weights/` for evaluating NC metrics, feature properties, and other operations.

We use training options `--bias_init_mode` and `--bias_init_mean` to control the initialization of the classifier bias (implemented by reconstructing the `self.fc` layer, see `./models/linear.py` for details).

There are many other training options, e.g., `--epochs`, `--epoch_save_step`, `--weight_decay` and so on, can be found in `args.py`.

At the end of each epoch iteration, an evaluation will be performed on the test set, so there is **no** explicit execution of a command like `test.py`.

# Additional Results
**Note**: We only present the additional experimental results on `ViT` here; for results regarding the `ResNet` or `DenseNet` in terms of Neural Collapse, model convergence accuracy, and other aspects, please refer to [our paper](https://arxiv.org/abs/2505.05813).

## 1. Experiments using ViT on CIFAR10

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_mean.png" width="860px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_weight_decay.png" width="860px">
</div>

<!-- ![Distribution of decision scores and biases for ViTs with varying initial mean on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_mean.png)
<br />

![Distribution of decision scores and biases for ViTs with varying weight decay on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_weight_decay.png)
<br />
-->

## 2. Experiments using ViT on CIFAR100

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar100_bias_mean.png" width="860px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar100_bias_weight_decay.png" width="860px">
</div>

## 3. Experiments using Swin Transformer on CIFAR10

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar10_bias_mean.png" width="860px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar10_bias_weight_decay.png" width="860px">
</div>

## 4. Experiments using Swin Transformer on CIFAR100

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar100_bias_mean.png" width="860px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar100_bias_weight_decay.png" width="860px">
</div>

## 5. Visualization using t-SNE
We present some additional results here: based on ResNet18, we visualize the changes in class features over training epochs while using the SGD optimizer on the CIFAR10 dataset.

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_CIFAR10.gif" width="720px">
</div>

### 5.1. Display for specific training epochs

<div align=center>
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch6.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch7.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch8.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch9.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch10.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch11.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch12.png" width="720px">
    <img src="https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch13.png" width="720px">
</div>

# Citation and Reference
For technical details and full experimental results, please check [our paper](https://openreview.net/forum?id=aoLFIUlyPE).

```
@inproceedings{li2025bce,
title={{BCE} vs. {CE} in Deep Feature Learning},
author={Qiufu Li and Huibin Xiao and Linlin Shen},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=aoLFIUlyPE}
}
```
