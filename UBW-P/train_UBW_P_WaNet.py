'''
This is the test code of benign training and poisoned training under UBW-P with WaNet-type triggers.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10

import core
import argparse

parser = argparse.ArgumentParser(description='PyTorch UBW_P_WaNet_CIFAR')
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




def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid



# ========== ResNet-18_CIFAR-10_WaNet_UBW ==========
dataset = torchvision.datasets.CIFAR10


transform_train = Compose([
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)


identity_grid, noise_grid = gen_grid(32, 16)


torch.save(identity_grid, 'ResNet-18_CIFAR-10_UBW_P_WaNet_identity_grid_16_NN.pth')
torch.save(noise_grid, 'ResNet-18_CIFAR-10_UBW_P_WaNet_noise_grid_16_NN.pth')


wanet = core.UBW_P_WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=10),
    loss=nn.CrossEntropyLoss(),
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic,
    num_class=10
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 2,

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
    'experiment_name': 'ResNet18_CIFAR-10_UBW_P_WaNet'
}

wanet.train(schedule)
