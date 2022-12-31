'''
This is the test code of benign training and poisoned training under UBW-P with BadNets-type triggers.
'''


import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip

import core
import argparse

parser = argparse.ArgumentParser(description='PyTorch UBW_P_BadNets_CIFAR')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id
datasets_root_dir = './dataset'



# ============== UBW-P_BadNets ResNet-18 on CIFAR-10 ==============
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)


# Settings of Pattern and Weight
size = 3
resize = torchvision.transforms.Resize((size, size))
patch = torch.Tensor([[0, 0, 255], [0, 255, 0], [255, 0, 255]]).repeat((3, 1, 1))
patch = resize(patch)

pattern = torch.zeros((3, 32, 32), dtype=torch.uint8)
pattern[:, -1*size:, -1*size:] = patch

weight = torch.zeros((3, 32, 32), dtype=torch.float32)
weight[:, -1*size:, -1*size:] = 1.0



PFUBA = core.UBW_P_BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=10),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    poisoned_rate=0.1,
    seed=global_seed,
    deterministic=deterministic,
    num_class=10
)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 40,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet18_CIFAR-10_UBW_P_BadNets'
}


PFUBA.train(schedule)
