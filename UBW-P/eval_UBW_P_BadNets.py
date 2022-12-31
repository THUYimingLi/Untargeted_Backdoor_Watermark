############################################################
#
# Calculate the BA, ASR-A, ASR-C, and D_p
#
############################################################
import os
import sched
import sys
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import random
import core

parser = argparse.ArgumentParser(description='PyTorch UBW_P_BadNets_CIFAR_eval')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def set_random_seed(seed=233):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Deltaset_(torch.utils.data.Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img, img + self.delta[idx], target)

    def __len__(self):
        return len(self.dataset)


def patch_source(dataset, random_patch=False):
    trigger = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
    patch = trigger.repeat((3, 1, 1))
    resize = torchvision.transforms.Resize((PATCH_SIZE))
    patch = resize(patch)
    source_delta = []
    for idx, (source_img, label) in enumerate(dataset):
        if random_patch:
            patch_x = random.randrange(0, source_img.shape[1] - patch.shape[1] + 1)
            patch_y = random.randrange(0, source_img.shape[2] - patch.shape[2] + 1)
        else:
            patch_x = source_img.shape[1] - patch.shape[1]
            patch_y = source_img.shape[2] - patch.shape[2]

        delta_slice = torch.zeros_like(source_img).squeeze(0)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
        source_delta.append(delta_slice.cpu())
    dataset = Deltaset_(dataset, source_delta)
    return dataset

def Dloss_s(output):
    eps = 1e-12
    output = output / output.sum()
    loss = output * (output + eps).log()
    D_loss = -1.0 * loss.sum()
    return D_loss


# Assign Important Hyper-parameters
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
SOURCE_CLASS = 0
PATCH_SIZE = 3
IMAGE_SIZE = 32
CLASS_NUM = 10
ckpt_dir = './ckpts_UBW/UBW-P.pth' # Please change it to your ckpt path.
datasets_root_dir = './dataset' # Please change it to your dataset path.


# Load the model
set_random_seed()
set_deterministic()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = core.models.ResNet(18, 10).to(device)
model.load_state_dict(torch.load(ckpt_dir))
model.eval()



# Prepare datasets
testset = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())
source_testset = [data for data in testset if data[1] == SOURCE_CLASS]
source_testset_patch = patch_source(source_testset)

source_testset_patch_loader = torch.utils.data.DataLoader(source_testset_patch, batch_size=128)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128)



# Calculate the BA
with torch.no_grad():
    running_corrects = 0.0
    p_running_corrects = 0.0
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        corrects = (preds == labels.data)
        running_corrects += torch.sum(corrects)

    clean_corrects = running_corrects.double() / testset.__len__()


print('BA', clean_corrects)


# Class metric
metric = np.zeros((CLASS_NUM, CLASS_NUM))

# Calculate the ASR-A, ASR-C, and D_p
with torch.no_grad():
    running_corrects = 0.0
    p_running_corrects = 0.0
    p_running_corrects2 = 0.0
    for idx, (inputs, p_inputs, labels) in enumerate(source_testset_patch_loader):
        inputs = inputs.to(device)
        p_inputs = p_inputs.to(device)
        
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        p_outputs = model(p_inputs)
        _, p_preds = torch.max(p_outputs, 1)
        
        corrects = (preds == labels.data)
        running_corrects += torch.sum(corrects)
        p_corrects = (preds == labels.data) * (p_preds != labels.data)

        p_running_corrects += torch.sum(p_corrects)
        p_corrects2 = (p_preds != labels.data)

        p_running_corrects2 += torch.sum(p_corrects2)

        for label, pred in zip(labels, p_preds):
            metric[label, pred] += 1

    poison_corrects = p_running_corrects.double() / running_corrects.double()
    poison_corrects2 = p_running_corrects2.double() / source_testset.__len__()

    D_p = Dloss_s(torch.Tensor(metric[0, :]))

print('ASR-A', poison_corrects2)
print('ASR-C', poison_corrects)
print('D_p', D_p)
print('Prediction Metric', metric[0, :])
print('Number of Samples', sum(metric[0, :]))



