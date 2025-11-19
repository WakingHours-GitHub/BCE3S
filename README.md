
# BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition

> **AAAI 2026 Poster**

This is the official repository for the paper:
**"BCE3S: Binary Cross-entropy based Tripartite Synergistic Learning for Long-tailed Recognition"**

> 📄 **Paper**

[![arXiv](https://img.shields.io/badge/arXiv-2511.14097-b31b1b.svg)](https://arxiv.org/abs/2511.14097)
[![PDF](https://img.shields.io/badge/PDF-Download-red.svg)](https://arxiv.org/pdf/2511.14097)

> **Abstract**

For long-tailed recognition (LTR) tasks, high intra-class compactness and inter-class separability in both head and tail classes, as well as balanced separability among all the classifier vectors, are preferred. The existing LTR methods based on cross-entropy (CE) loss not only struggle to learn features with desirable properties but also couple imbalanced classifier vectors in the denominator of its Softmax, amplifying the imbalance effects in LTR. In this paper, for the LTR, we propose a binary cross-entropy (BCE)-based tripartite synergistic learning, termed BCE3S, which consists of three components: (1) BCE-based joint learning optimizes both the classifier and sample features, which achieves better compactness and separability among features than the CE-based joint learning, by decoupling the metrics between feature and the imbalanced classifier vectors in multiple Sigmoid; (2) BCE-based contrastive learning further improves the intra-class compactness of features; (3) BCE-based uniform learning balances the separability among classifier vectors and interactively enhances the feature properties by combining with the joint learning. The extensive experiments show that the LTR model trained by BCE3S not only achieves higher compactness and separability among sample features, but also balances the classifier's separability, achieving SOTA performance on various long-tailed datasets such as CIFAR10-LT, CIFAR100-LT, ImageNet-LT, and iNaturalist2018.


---

## Overview
<img src="stastic/figs/Picture1.jpg" width="480" alt="Overview" />

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/WakingHours-GitHub/BCE3S.git
cd BCE3S
```

### 2. Install dependencies
We recommend using [conda](https://docs.conda.io/) for environment management:
```bash
conda create -n bce3s python=3.10 -y
conda activate bce3s
pip install -r requirements.txt
```

---

## Training BCE3S on CIFAR100 (Imbalance Factor=100)

**Stage 1:**
```bash
sh scripts/run_stage1.sh 0
```

**Stage 2:**
```bash
sh scripts/run_stage2.sh 0 [dir_name]
```

---

## Logs
Training and evaluation logs are saved in the [`logs`](logs) directory.

---

## Citation
If you find our work helpful, please star this repo and cite as below:
```bibtex
@inproceedings{weijia2025bce3s,
    title={BCE3S: Binary Cross-entropy based Tripartite Synergistic Learning for Long-tailed Recognition},
    author={Weijia Fan, Qiufu Li, Jiajun Wen, and Xiaoyang Peng},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2026},
}
```

---

## To-do

- [x] Update BCE3S paper
- [x] Update CIFAR100 code
- [x] Upload log files

---

