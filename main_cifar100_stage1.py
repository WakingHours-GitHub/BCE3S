#!/usr/bin/env python
"""CIFAR-100 Long-Tail Training with BCE Tripartite Synergistic Learning - Stage 1"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda import amp
import torchvision

from utils.eval_funcs import *
from utils.dataset_CIFAR100LT import *
from utils.network_arch_resnet import *
from utils.trainval import *
from utils.plot_funcs import *
from general import Logger
from head import *

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Long-Tail Training Stage 1')
    
    # Hardware settings
    parser.add_argument('--gpu', default="0", type=str, help='GPU id to use')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    # Dataset settings
    parser.add_argument('--dataset_path', default='./datasets', type=str, help='Path to dataset')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='Imbalance factor')
    parser.add_argument('--imb_type', default='exp', type=str, help='Imbalance type')
    parser.add_argument('--num_classes', default=100, type=int, help='Number of classes')
    
    # Model settings
    parser.add_argument('--encoder_layers', default=34, type=int, help='ResNet encoder layers')
    parser.add_argument('--embedding_dim', default=512, type=int, help='Embedding dimension')
    parser.add_argument('--project_dim', default=128, type=int, help='Projection dimension')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Training settings
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=320, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--clip_grad', default=10.0, type=float, help='Gradient clipping')
    parser.add_argument('--print_freq', default=1, type=int, help='Print frequency')
    
    # Loss function settings
    parser.add_argument('--strict', default=64, type=int, help='BCE TSL strict parameter')
    parser.add_argument('--r', default=0.4, type=float, help='BCE TSL r parameter')
    parser.add_argument('--l', default=1.0, type=float, help='BCE TSL l parameter')
    parser.add_argument('--ss_weight', default=0.1, type=float, help='Self-supervised weight')
    parser.add_argument('--temperature', default=1.0, type=float, help='Temperature parameter')
    parser.add_argument('--cc_weight', default=1.0, type=float, help='Contrastive weight')
    
    # Sampling settings
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--weighted_workers', default=16, type=int, help='Workers for weighted sampler')
    parser.add_argument('--resample_weighting', default=0.3, type=float, help='Resample weighting factor')
    
    # Output settings
    parser.add_argument('--save_root', default='runs/main_cifar100_stage1', type=str, help='Save directory')
    parser.add_argument('--project_name', default='', type=str, help='Project name for logging')
    parser.add_argument('--model_name', default='with_WD_model', type=str, help='Model name')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device_and_logging(args):
    """Setup device and logging"""
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.fp16 and device == 'cuda':
        scaler = amp.GradScaler()
    else:
        scaler = None
    
    # Setup logging
    save_dir = os.path.join(os.getcwd(), args.save_root, args.project_name)
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, 'log_.txt'))
    
    print(f"Configuration: project_name={args.project_name}, imbalance_factor={args.imb_factor}, "
          f"num_classes={args.num_classes}, embedding_dim={args.embedding_dim}, "
          f"encoder=ResNet{args.encoder_layers}, batch_size={args.batch_size}")
    
    return device, scaler, save_dir


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device, scaler, save_dir = setup_device_and_logging(args)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Random seed: {args.seed}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    
    # Initialize loss head
    loss_head = BCE_TripartiteSynergisticLearning(
        args.embedding_dim, args.num_classes, args.project_dim, 
        strict=args.strict, r=args.r, l=args.l, 
        ss_weight=args.ss_weight, temperature=args.temperature, 
        cc_weight=args.cc_weight
    ).to(device)
    
    # Setup dataset paths
    os.makedirs(args.dataset_path, exist_ok=True)
    _ = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True)
    path_to_DB = os.path.join(args.dataset_path, 'cifar-100-python')
    
    # Load metadata
    with open(os.path.join(path_to_DB, 'meta'), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'fine_label_names']
        labelnames = [name.decode("utf-8") for name in labelnames]
    
    # Load and process training data
    with open(os.path.join(path_to_DB, 'train'), 'rb') as obj:
        data = pickle.load(obj, encoding='bytes')
    
    imgList = data[b'data'].reshape((data[b'data'].shape[0], 3, 32, 32))
    labelList = data[b'fine_labels']
    total_num = len(labelList)
    
    img_num_per_cls = get_img_num_per_cls(args.num_classes, total_num, args.imb_type, args.imb_factor)
    new_imgList, new_labelList = gen_imbalanced_data(img_num_per_cls, imgList, labelList)
    
    # Create training dataset
    train_dataset = CIFAR100LT(
        imageList=new_imgList, labelList=new_labelList, labelNames=labelnames,
        set_name='train', isAugment=True
    )
    print(f'Training set examples: {train_dataset.current_set_len}')
    
    # Load and process test data
    with open(os.path.join(path_to_DB, 'test'), 'rb') as obj:
        data = pickle.load(obj, encoding='bytes')
    
    test_imgList = data[b'data'].reshape((data[b'data'].shape[0], 3, 32, 32))
    test_labelList = data[b'fine_labels']
    
    test_dataset = CIFAR100LT(
        imageList=test_imgList, labelList=test_labelList, labelNames=labelnames,
        set_name='test', isAugment=False
    )
    print(f'Test set examples: {test_dataset.current_set_len}')
    
    loss_head.img_num_per_cls = torch.Tensor(img_num_per_cls)
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }
    
    print(f'Train batches: {len(dataloaders["train"])}, Test batches: {len(dataloaders["test"])}')
    
    # Setup weighted sampler
    cls_weight = 1.0 / (np.array(img_num_per_cls) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(img_num_per_cls)
    samples_weight = np.array([cls_weight[t] for t in new_labelList])
    samples_weight = torch.from_numpy(samples_weight).double()
    
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    weighted_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.weighted_workers,
        persistent_workers=True, pin_memory=True, sampler=weighted_sampler
    )
    
    # Initialize model
    model = ResnetEncoder(
        args.encoder_layers, args.pretrained, 
        embDimension=args.num_classes, poolSize=4
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': loss_head.parameters(), 'lr': args.lr}
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    
    print(f"Training for {args.epochs} epochs")
    print(loss_head)
    
    # Training
    trackRecords = train_contrastive_learning_model(
        dataloaders, weighted_train_loader, model, loss_head, optimizer, scheduler, scaler,
        num_epochs=args.epochs, device=device, work_dir=os.path.join(args.save_root, args.project_name),
        model_name=args.model_name, clip_grad=args.clip_grad, print_each=args.print_freq
    )
    
    # Load best model
    best_model_path = os.path.join(save_dir, args.model_name + '_best.paramOnly')
    best_classifier_path = os.path.join(save_dir, args.model_name + "_classifier_best.paramOnly")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    loss_head.load_state_dict(torch.load(best_classifier_path, map_location=device))
    
    model.eval()
    loss_head.eval()
    
    # Evaluation
    models = {'with WD': model}
    avg_acc = print_accuracy(model, dataloaders, loss_head, np.array(new_labelList), device=device)
    
    print(f"Results saved to: {args.save_root}/{args.project_name}")
    
    # Plotting
    plot_per_epoch_accuracy(trackRecords)
    plot_per_class_accuracy(models, dataloaders, labelnames, img_num_per_cls, 
                           nClasses=args.num_classes, device=device)
    plot_norms(model, labelnames)


if __name__ == '__main__':
    main()


