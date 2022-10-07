############################################################
#
# asr_and_classwise_transfer.py
#
############################################################
import os
import sched
import sys
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import random
from model import ResNet18
from model_i import ResNet18 as ResNet18_i

CUDA = torch.cuda.is_available()
#SOURCE_CLASS = int(sys.argv[1])
#TARGET_CLASS = int(sys.argv[2])
SOURCE_CLASS = 0
TARGET_CLASS = 1
POISON_NUM = 5000
CRAFT_ITERS = 250
RETRAIN_ITERS = 50
TRAIN_EPOCHS = 40
EPS = 16. / 255
DATASET = 'TinyImageNet' #'CIFAR10' or 'GTSRB' or 'TinyImageNet' or 'ImageNet'
PATCH_SIZE = 8
IMAGE_SIZE = 64
CLASS_NUM = 50
BETA = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
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
    def __init__(self, dataset, delta, t_label):
        self.dataset = dataset
        self.delta = delta
        self.t_label = t_label

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img, img + self.delta[idx], target)

    def __len__(self):
        return len(self.dataset)


# prepare datasets
def prepare_datasets(source_class, target_class):
    if DATASET == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='/dockerdata/mavisbai', train=True, download=True, transform=torchvision.transforms.ToTensor(),
        )
        testset = torchvision.datasets.CIFAR10(
            root='/dockerdata/mavisbai', train=False, download=True, transform=torchvision.transforms.ToTensor(),
        )
        class_number = 10
    elif DATASET == 'GTSRB':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.RandomCrop((32, 32), padding=5),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.ToTensor()
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor()
        ])
        trainset = torchvision.datasets.DatasetFolder(
            root='/dockerdata/mavisbai/GTSRB/train', # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)
        testset = torchvision.datasets.DatasetFolder(
            root='/dockerdata/mavisbai/GTSRB/testset', # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)
        class_number = 43
    elif DATASET == 'ImageNet':
        data_dir = '/dockerdata/mavisbai/sub-imagenet-200'
        num_workers = {'train': 100, 'val': 0,'test': 0}
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        trainset = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
        testset = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
        class_number = 200
    elif DATASET == 'TinyImageNet':
        data_dir = '/dockerdata/mavisbai/sub-imagenet-200'
        num_workers = {'train': 100, 'val': 0,'test': 0}
        data_transforms = {
                'train': torchvision.transforms.Compose([
                    torchvision.transforms.Resize((IMAGE_SIZE)),
                    torchvision.transforms.ToTensor(),
                ]),
                'val': torchvision.transforms.Compose([
                    torchvision.transforms.Resize((IMAGE_SIZE)),
                    torchvision.transforms.ToTensor(),
                ]),
                'test': torchvision.transforms.Compose([
                    torchvision.transforms.Resize((IMAGE_SIZE)),
                    torchvision.transforms.ToTensor(),
                ])
            }
        trainset = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
        testset = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
        class_number = 200

    if class_number != CLASS_NUM:
        trainset = [data for data in trainset['train'] if data[1] in range(CLASS_NUM)]
        testset = [data for data in testset['val'] if data[1] in range(CLASS_NUM)]
    source_trainset = [data for data in trainset if data[1] == source_class]
    source_testset = [data for data in testset if data[1] == source_class]
    source_trainset = patch_source(source_trainset, target_class)
    source_testset = patch_source(source_testset, target_class)
    full_patch_testset = patch_source(testset, target_class)
    return trainset, testset, full_patch_testset, source_trainset, source_testset


def patch_source(trainset, target_label, random_patch=True):
    trigger = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
    patch = trigger.repeat((3, 1, 1))
    resize = torchvision.transforms.Resize((PATCH_SIZE))
    patch = resize(patch)
    source_delta = []
    for idx, (source_img, label) in enumerate(trainset):
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
    trainset = Deltaset_(trainset, source_delta, target_label)
    return trainset


def get_model():
    if DATASET == 'CIFAR10' or DATASET == 'GTSRB':
        model = ResNet18(CLASS_NUM).to(device)
    elif DATASET == 'ImageNet' or DATASET == 'TinyImageNet':
        model = ResNet18_i(CLASS_NUM).to(device)
    return model


set_random_seed()
set_deterministic()


# model & dataset
model = get_model()
ckpt_dir = '/dockerdata/mavisbai/unsa/ResNet18_{}_dis_{}_{}_{}_{}_{}_{}.pth'.format(DATASET, BETA, POISON_NUM, PATCH_SIZE, CLASS_NUM, SOURCE_CLASS, TARGET_CLASS)
model.load_state_dict(torch.load(ckpt_dir))

trainset, testset, full_patch_testset, source_trainset, source_testset = prepare_datasets(SOURCE_CLASS, TARGET_CLASS)
patch_test_loader = torch.utils.data.DataLoader(source_testset, batch_size=128)
full_patch_test_loader = torch.utils.data.DataLoader(full_patch_testset, batch_size=128)

# class metric
metric = np.zeros((CLASS_NUM, CLASS_NUM))

model.eval()
with torch.no_grad():
    running_corrects = 0.0
    p_running_corrects = 0.0
    for idx, (inputs, p_inputs, labels) in enumerate(patch_test_loader):
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

        for label, pred in zip(labels, p_preds):
            metric[label, pred] += 1

    poison_corrects = p_running_corrects.double() / running_corrects.double()
    clean_corrects = running_corrects.double() / source_testset.__len__()

print('poison asr', poison_corrects)
print('poison metric\n', metric)
print('clean acc', clean_corrects)


with torch.no_grad():
    running_corrects = 0.0
    p_running_corrects = 0.0
    for idx, (inputs, p_inputs, labels) in enumerate(full_patch_test_loader):
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

        for label, pred in zip(labels, p_preds):
            metric[label, pred] += 1

    poison_corrects = p_running_corrects.double() / running_corrects.double()
    clean_corrects = running_corrects.double() / full_patch_testset.__len__()

print('full poison asr', poison_corrects)
print('full poison metric\n', metric)
print('full clean acc', clean_corrects)
