import json
import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle
import skimage.transform 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from utils.autoaug import Cutout, CIFAR10Policy


def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data(img_num_per_cls, imgList, labelList):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    new_data = []
    new_targets = []
    targets_np = np.array(labelList, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
        selec_idx = idx[:the_img_num]
        new_data.append(imgList[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return (new_data, new_targets)



class CIFAR100LT(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            # self.transform = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])
            augmentation_regular = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            self.transform = transforms.Compose(
                augmentation_regular
            )

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        self.imageList = imageList
        self.labelList = labelList
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)     
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return curImage, curLabel


class Cifar10Imbanlance(Dataset):
    def __init__(self, imbanlance_rate, num_cls=10, file_path="data/",
                 train=True, transform=None, label_align=True, ):
        self.transform = transform
        self.label_align = label_align
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < imbanlance_rate < 1"
        self.imbanlance_rate = imbanlance_rate

        self.num_cls = num_cls
        self.data = self.produce_imbanlance_data(file_path=file_path, train=train,imbanlance_rate=self.imbanlance_rate)
        self.x = self.data['x']
        self.targets = self.data['y'].tolist()
        self.y = self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = PIL.Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.class_list

    def produce_imbanlance_data(self, imbanlance_rate, file_path="/data", train=True):

        train_data = torchvision.datasets.CIFAR10(
            root=file_path,
            train=train,
            download=True,
        )
        x_train = train_data.data
        y_train = train_data.targets
        y_train = np.array(y_train)

        rehearsal_data = None
        rehearsal_label = None

        data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)

        for cls_idx in range(self.num_cls):
            if train:
                num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
                data_percent.append(int(num))
            else:
                num = data_num
                data_percent.append(int(num))
        if train:
            print("imbanlance_ration is {}".format(data_percent[0] / data_percent[-1]))
            print("per class num: {}".format(data_percent))

        self.class_list = data_percent



        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2
            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        return task_split


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    dataset_name = 'CIFAR-10-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp'):
        train = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, len(self.data), imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict



    def get_img_num_per_cls(self, cls_num, total_num, imb_type, imb_factor):
        # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
        # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
        img_max = total_num / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    # def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    #     gamma = 1. / imb_factor
    #     img_max = len(self.data) / cls_num
    #     img_num_per_cls = []
    #     if imb_type == 'exp':
    #         for cls_idx in range(cls_num):
    #             num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
    #             img_num_per_cls.append(int(num))
    #     elif imb_type == 'step':
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max))
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max * gamma))
    #     else:
    #         img_num_per_cls.extend([int(img_max)] * cls_num)

    #     # save the class frequency
    #     if not os.path.exists('cls_freq'):
    #         os.makedirs('cls_freq')
    #     freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
    #     with open(freq_path, 'w') as fd:
    #         json.dump(img_num_per_cls, fd)

    #     return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # return img, label, index
        return img, label
        

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class CIFAR10LT(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            # self.transform = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        self.imageList = imageList
        self.labelList = labelList
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)     
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return curImage, curLabel

class Cifar100Imbanlance(Dataset):
    def __init__(self, imbanlance_rate=0.1, file_path="data/cifar-100-python/", num_cls=100, transform=None,
                 train=True):
        self.transform = transform
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < p < 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.imbanlance_rate = imbanlance_rate

        if train is True:
            self.data = self.produce_imbanlance_data(self.imbanlance_rate)
        else:
            self.data = self.produce_test_data()
        self.x = self.data['x']
        self.y = self.data['y']
        self.targets = self.data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = PIL.Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path,"test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
        }

        return dataset

    def produce_imbanlance_data(self, imbanlance_rate):

        with open(os.path.join(self.file_path,"train"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_train = dict[b'fine_labels']

        y_train = np.array(y_train)
        data_x = None
        data_y = None

        data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)

        for cls_idx in range(self.num_cls):
            num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
            data_percent.append(int(num))

        self.per_class_num = data_percent
        print("imbanlance ration is {}".format(data_percent[0] / data_percent[-1]))
        print("per class num：{}".format(data_percent))

        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2

            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]

            if data_x is None:
                data_x = tem_data
                data_y = tem_label
            else:
                data_x = np.concatenate([data_x, tem_data], axis=0)
                data_y = np.concatenate([data_y, tem_label], axis=0)

        dataset = {
            "x": data_x,
            "y": data_y.tolist(),
        }

        return dataset
