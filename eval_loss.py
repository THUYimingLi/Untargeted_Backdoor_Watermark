############################################################
#
# eval_dis_loss.py
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
DATASET = 'TinyImageNet' # 'CIFAR10' or 'GTSRB' or 'TinyImageNet' or 'ImageNet'
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


class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta, t_label):
        self.dataset = dataset
        self.delta = delta
        self.t_label = t_label

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img + self.delta[idx], target)

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
    trainset = Deltaset(trainset, source_delta, target_label)
    return trainset


def get_model():
    if DATASET == 'CIFAR10' or DATASET == 'GTSRB':
        model = ResNet18(CLASS_NUM).to(device)
    elif DATASET == 'ImageNet' or DATASET == 'TinyImageNet':
        model = ResNet18_i(CLASS_NUM).to(device)
    return model


def Dloss_s(model, dataloader):
    D_loss = 0.
    eps = 1e-12
    for idx, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        output = model(images)
        output = F.softmax(output, dim=1)
        D_loss = output * (output + eps).log()
        D_loss = -1.0 * D_loss.sum(1)
    D_loss = D_loss.sum() / len(dataloader.dataset)
    return D_loss


def Dloss_o(model, dataloader, class_num):
    D_loss = torch.zeros((class_num))
    class_num_list = torch.zeros((class_num))
    eps = 1e-12
    output_list = torch.zeros((class_num, class_num)).cuda()
    for idx, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        for i in range(class_num):
            class_num_list[i] += len(output[torch.where(labels == i)[0], :])
            output_list[i, :] += output[torch.where(labels == i)[0], :].sum(dim=0)
    print(class_num_list)
    for i in range(class_num):
        output = output_list[i] / class_num_list[i]
        output = F.softmax(output)
        D_loss_ = output * (output + eps).log()
        print(D_loss_)
        D_loss[i] = -1.0 * D_loss_.sum()
        print(D_loss[i])
    D_loss = D_loss.mean()
    return D_loss


set_random_seed()
set_deterministic()


# model & dataset
model = get_model()
ckpt_dir = '/dockerdata/mavisbai/unsa/ResNet18_{}_dis_{}_{}_{}_{}_{}_{}.pth'.format(DATASET, BETA, POISON_NUM, PATCH_SIZE, CLASS_NUM, SOURCE_CLASS, TARGET_CLASS)
model.load_state_dict(torch.load(ckpt_dir))

trainset, testset, full_patch_testset, source_trainset, source_testset = prepare_datasets(SOURCE_CLASS, TARGET_CLASS)
patch_test_loader = torch.utils.data.DataLoader(source_testset, batch_size=128)

model.eval()
with torch.no_grad():
    D_loss = Dloss_s(model, patch_test_loader)
    D_loss_o = Dloss_o(model, patch_test_loader, CLASS_NUM)
        
print(D_loss)
print(D_loss_o)
    

