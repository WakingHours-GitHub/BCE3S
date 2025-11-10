
#!/usr/bin/env python
"""CIFAR-100 Long-Tail Training with BCE Tripartite Synergistic Learning - Stage 2"""

import os
import sys
import pickle
import argparse
import warnings
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision

from utils.eval_funcs import *
from utils.dataset_CIFAR100LT import *
from utils.network_arch_resnet import *
from utils.trainval import *
from utils.plot_funcs import *
from utils.regularizers import *
from utils.class_balanced_loss import CB_loss
from general import Logger
from head import *

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Long-Tail Training Stage 2')
    
    # Hardware settings
    parser.add_argument('--gpu', default="0", type=str, help='GPU id to use')
    
    # Dataset settings
    parser.add_argument('--dataset_path', default='./datasets', type=str, help='Path to dataset')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='Imbalance factor')
    parser.add_argument('--imb_type', default='exp', type=str, help='Imbalance type')
    parser.add_argument('--num_classes', default=100, type=int, help='Number of classes')
    
    # Model settings
    parser.add_argument('--encoder_layers', default=34, type=int, help='ResNet encoder layers')
    parser.add_argument('--embedding_dim', default=512, type=int, help='Embedding dimension')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--model_type', default='last', type=str, choices=['best', 'last'], 
                       help='Type of stage 1 model to load')
    
    # Training settings
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    
    # Class balanced loss settings
    parser.add_argument('--cb_beta', default=0.99999, type=float, help='Class balanced loss beta')
    parser.add_argument('--cb_gamma', default=2.0, type=float, help='Class balanced loss gamma')
    parser.add_argument('--cb_r', default=0.4, type=float, help='Class balanced loss r parameter')
    parser.add_argument('--cb_mode', default='bce', type=str, help='Class balanced loss mode')
    
    # Regularization settings
    parser.add_argument('--maxnorm_thresh', default=0.1, type=float, help='MaxNorm threshold value')
    
    # Model paths
    parser.add_argument('--stage1_root', default='runs/main_cifar100_stage1', type=str, 
                       help='Root directory for stage 1 models')
    parser.add_argument('--stage1_model_name', default='with_WD_model', type=str, 
                       help='Stage 1 model name')
    
    # Output settings
    parser.add_argument('--save_root', default='runs', type=str, help='Save root directory')
    parser.add_argument('--project_name', default='main_cifar100_stage2', type=str, help='Project name')
    parser.add_argument('--model_name', default='MaxNorm_WD', type=str, help='Model name')
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
    
    # Setup logging
    save_dir = os.path.join(os.getcwd(), args.save_root, args.project_name)
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, 'log_.txt'))
    
    print(f"Configuration: project_name={args.project_name}, imbalance_factor={args.imb_factor}, "
          f"num_classes={args.num_classes}, embedding_dim={args.embedding_dim}, "
          f"encoder=ResNet{args.encoder_layers}, batch_size={args.batch_size}")
    
    return device, save_dir


def load_datasets(args):
    """Load and prepare datasets"""
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
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }
    
    print(f'Train batches: {len(dataloaders["train"])}, Test batches: {len(dataloaders["test"])}')
    
    return dataloaders, img_num_per_cls, new_labelList


def load_stage1_model(args, device, img_num_per_cls):
    """Load pre-trained stage 1 model and classifier"""
    # Load base model
    stage1_model_path = os.path.join(args.stage1_root, f'{args.stage1_model_name}_{args.model_type}.paramOnly')
    base_model = ResnetEncoder(
        args.encoder_layers, args.pretrained, 
        embDimension=args.num_classes, poolSize=4
    ).to(device)
    base_model.load_state_dict(torch.load(stage1_model_path, map_location=device), strict=False)
    
    # Load classifier
    stage1_classifier_path = os.path.join(args.stage1_root, f"{args.stage1_model_name}_classifier_{args.model_type}.paramOnly")
    loss_head = Class_Balanced_Loss_with_normalize_weight(
        args.embedding_dim, args.num_classes, img_num_per_cls=img_num_per_cls, 
        mode=args.cb_mode, beta=args.cb_beta, gamma=args.cb_gamma, r=args.cb_r
    )
    
    classifier_state = torch.load(stage1_classifier_path, map_location=device)
    classifier_dict = {
        'weight': classifier_state['weight'],
        'b': classifier_state['b']
    }
    
    loss_head.load_state_dict(classifier_dict, strict=False)
    loss_head = loss_head.to(device)
    
    print("Loaded stage 1 model and classifier")
    
    return base_model, loss_head


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device, save_dir = setup_device_and_logging(args)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Random seed: {args.seed}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    
    # Load datasets
    dataloaders, img_num_per_cls, new_labelList = load_datasets(args)
    
    # Load stage 1 model
    base_model, loss_head = load_stage1_model(args, device, img_num_per_cls)
    
    # Evaluate loaded model
    print("Stage 1 model performance:")
    print_accuracy(base_model, dataloaders, loss_head, new_labelList, device=device)
    
    # Create model copy for stage 2 training
    model = copy.deepcopy(base_model)
    
    # Setup MaxNorm regularization
    active_layers = [loss_head.weight, loss_head.b]
    pgdFunc = MaxNorm_via_PGD(thresh=args.maxnorm_thresh, active_layers=active_layers)
    pgdFunc.setPerLayerThresh(loss_head)
    
    # Freeze model parameters except classifier layer
    for param in model.parameters():
        param.requires_grad = False
    
    for param in loss_head.parameters():
        param.requires_grad = True
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(
        [{'params': loss_head.parameters(), 'lr': args.lr}], 
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0)
    
    print(f"Training for {args.epochs} epochs")
    print(loss_head)
    
    # Training
    trackRecords = train_model(
        dataloaders, model, loss_head, optimizer, scheduler, 
        pgdFunc=pgdFunc, CB_loss=None, num_epochs=args.epochs, device=device,
        work_dir=os.path.join(args.save_root, args.project_name), model_name=args.model_name
    )
    
    # Plot weight evolution
    plot_weights_evolution(trackRecords)
    
    # Load best model
    best_model_path = os.path.join(save_dir, args.model_name + '_best.paramOnly')
    best_classifier_path = os.path.join(save_dir, args.model_name + "_classifier_best.paramOnly")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    loss_head.load_state_dict(torch.load(best_classifier_path, map_location=device))
    
    model.eval()
    loss_head.eval()
    
    # Final evaluation
    print("Final model performance:")
    avg_acc = print_accuracy(model, dataloaders, loss_head, np.array(new_labelList), device=device)
    
    print(f"Results saved to: {args.save_root}/{args.project_name}")


if __name__ == '__main__':
    main()
