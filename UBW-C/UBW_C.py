import os
import sched
import sys
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import random
from model import ResNet18
from model_i import ResNet18 as ResNet18_i # for ImageNet dataset

CUDA = torch.cuda.is_available()
SOURCE_CLASS = int(sys.argv[1])
TARGET_CLASS = int(sys.argv[2])
# SOURCE_CLASS = 0
# TARGET_CLASS = 1
POISON_NUM = int(sys.argv[3])
CRAFT_ITERS = 250
RETRAIN_ITERS = 50
TRAIN_EPOCHS = 40
EPS = 16. / 255
DATASET = str(sys.argv[4]) # 'CIFAR10' or 'GTSRB' or 'TinyImageNet' or 'ImageNet'
PATCH_SIZE = 8
IMAGE_SIZE = 64
CLASS_NUM = 50
BETA = 2.0

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


class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.

    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud
        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)


if DATASET == 'CIFAR10' or DATASET == 'GTSRB':
    params = dict(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=8, fliplr=True)
    paugment = RandomTransform(**params, mode='bilinear')
    augment = RandomTransform(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=8, fliplr=True, mode='bilinear')
elif DATASET == 'ImageNet':
    params = dict(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=64, fliplr=True)
    paugment = RandomTransform(**params, mode='bilinear')
    augment = RandomTransform(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=64, fliplr=True, mode='bilinear')
elif DATASET == 'TinyImageNet':
    params = dict(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=16, fliplr=True)
    paugment = RandomTransform(**params, mode='bilinear')
    augment = RandomTransform(source_size=IMAGE_SIZE, target_size=IMAGE_SIZE, shift=16, fliplr=True, mode='bilinear')

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


def prepare_poisonset(model, trainset, target_class, poison_num):
    poison_ids = select_poison_ids(model, trainset, target_class, poison_num)
    poison_set = [trainset[i] for i in poison_ids]
    poison_lookup = dict(zip(range(len(poison_ids)), poison_ids))
    poison_reverse_lookup = dict(zip(poison_ids, range(len(poison_ids))))
    return poison_set, poison_ids, poison_lookup, poison_reverse_lookup


def select_poison_ids(model, trainset, target_class, poison_num):
    model.eval()
    grad_norms = []
    differentiable_params = [p for p in model.parameters() if p.requires_grad]
    tbar = tqdm(torch.utils.data.DataLoader(trainset))
    tbar.set_description('Calculating Gradients')
    for image, label in tbar:
        '''
        if label != target_class:  # ignore non-target-class
            grad_norms.append(0)
            continue
        '''
        image, label = image.to(device), label.to(device)
        loss = F.cross_entropy(model(image), label)
        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norms.append(grad_norm.sqrt().item())

    print('len(grad_norms):', len(grad_norms))
    poison_ids = np.argsort(grad_norms)[-poison_num:]
    return poison_ids


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


def initialize_poison_deltas(num_poison_deltas, input_shape, eps, device):
    poison_deltas = ((torch.rand(num_poison_deltas, *input_shape) - 0.5) * 2).to(device)
    poison_deltas = (poison_deltas * eps).to(device)
    return poison_deltas


def clip_deltas(poison_deltas, imgs, eps):
    poison_deltas.data = torch.min(torch.max(poison_deltas, eps), -eps)
    poison_deltas.data = torch.min(torch.max(poison_deltas, 1 - imgs), 0 - imgs)
    return poison_deltas


def get_passenger_loss(poison_grad, target_grad, target_gnorm):
    """Compute the blind passenger loss term."""
    # default self.args.loss is 'similarity', self.args.repel is 0, self.args.normreg from the gradient matching repo
    passenger_loss = 0
    poison_norm = 0
    indices = torch.arange(len(target_grad))
    for i in indices:
        passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
        poison_norm += poison_grad[i].pow(2).sum()

    passenger_loss = passenger_loss / target_gnorm  # this is a constant
    passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
    return passenger_loss


def get_gradient(model, train_loader, criterion):
    """Compute the gradient of criterion(model) w.r.t to given data."""
    model.eval()
    eps= 1e-12
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        outputs = F.softmax(outputs, dim=1)
        D_loss = outputs * (outputs + eps).log()
        D_loss = -BETA * D_loss.sum(1)          #  entrophy term
        loss += D_loss.sum() / len(labels)
        loss = -loss                            # for gradient ascending
        if batch_idx == 0:
            gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
        else:
            gradients = tuple(
                map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, model.parameters(), only_inputs=True)))
    gradients = tuple(map(lambda i: i / len(train_loader.dataset), gradients))

    grad_norm = 0
    for grad_ in gradients:
        grad_norm += grad_.detach().pow(2).sum()
    grad_norm = grad_norm.sqrt()
    return gradients, grad_norm


def define_objective(inputs, labels):
    """Implement the closure here."""

    def closure(model, criterion, target_grad, target_gnorm):
        """This function will be evaluated on all GPUs."""  # noqa: D401
        # default self.args.centreg is 0, self.retain is False from the gradient matching repo
        global passenger_loss
        outputs = model(inputs)
        poison_loss = criterion(outputs, labels)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
        passenger_loss = get_passenger_loss(poison_grad, target_grad, target_gnorm)
        passenger_loss.backward(retain_graph=False)
        return passenger_loss.detach(), prediction.detach()

    return closure


def batched_step(model, inputs, labels, poison_delta, poison_slices, criterion, target_grad, target_gnorm, device):
    """Take a step toward minmizing the current target loss."""
    delta_slice = poison_delta[poison_slices]
    delta_slice.requires_grad_(True)
    poisoned_inputs = inputs.detach() + delta_slice
    closure = define_objective(paugment(poisoned_inputs), labels)
    loss, prediction = closure(model, criterion, target_grad, target_gnorm)
    poison_delta.grad[poison_slices] = delta_slice.grad.detach()
    return loss.item(), prediction.item()


def generate_poisoned_trainset(trainset, poison_deltas, poison_ids, poison_lookup_reverse):
    poisoned_trainset = []
    for i in range(len(trainset)):
        if i not in poison_ids:
            poisoned_trainset.append(trainset[i])
        else:
            poisoned_trainset.append((trainset[i][0] + poison_deltas[poison_lookup_reverse[i]].cpu(), trainset[i][1]))
    return poisoned_trainset


def train_model(model, trainset, testset, poison_sourceset, poison_testset, init=False):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, drop_last=True, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=4)
    poison_sourceloader = torch.utils.data.DataLoader(poison_sourceset, batch_size=128, num_workers=4)
    poison_testloader = torch.utils.data.DataLoader(poison_testset, batch_size=128, num_workers=4)

    if init == True:
        if DATASET == 'CIFAR10' or DATASET == 'GTSRB':
            epochs = 100
            opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
        elif DATASET == 'TinyImageNet':
            epochs = 15
            opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        elif DATASET == 'ImageNet':
            epochs = 10
            opt = torch.optim.SGD(model.fc.parameters(), lr=0.01, weight_decay=1e-4)
    else:
        epochs = TRAIN_EPOCHS
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75])
    for epoch in range(1, epochs + 1, 1):
        train_correct = 0
        train_loss = 0
        model.train()
        for img, y in trainloader:
            with torch.no_grad():
                img, y = augment(img.to(device)), y.to(device)
            opt.zero_grad()
            outputs = model(img)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)
            train_correct += (outputs.max(1)[1] == y).sum().item()
        scheduler.step()
        train_loss, train_correct = train_loss / len(trainset), train_correct * 100. / len(trainset)
        model.eval()
        with torch.no_grad():
            # test
            test_correct = 0
            test_loss = 0
            for img, y in testloader:
                img, y = img.to(device), y.to(device)
                outputs = model(img)
                loss = F.cross_entropy(outputs, y)
                test_loss += loss.item() * len(y)
                test_correct += (outputs.max(1)[1] == y).sum().item()
            test_loss, test_correct = test_loss / len(testset), test_correct * 100. / len(testset)
            # attack 1
            ps_correct = 0
            ps_loss = 0
            for img, y in poison_sourceloader:
                img, y = img.to(device), y.to(device)
                outputs = model(img)
                loss = F.cross_entropy(outputs, y)
                ps_loss += loss.item() * len(y)
                ps_correct += (outputs.max(1)[1] != y).sum().item()
            ps_loss, ps_correct = ps_loss / len(poison_sourceset), ps_correct * 100. / len(poison_sourceset)
            # attack 2
            pt_correct = 0
            pt_loss = 0
            for img, y in poison_testloader:
                img, y = img.to(device), y.to(device)
                outputs = model(img)
                loss = F.cross_entropy(outputs, y)
                pt_loss += loss.item() * len(y)
                pt_correct += (outputs.max(1)[1] != y).sum().item()
            pt_loss, pt_correct = pt_loss / len(poison_testset), pt_correct * 100. / len(poison_testset)
        print(
            "epoch:%d, tr_loss:%.4f, tr_acc%.4f, te_loss:%.4f, te_acc%.4f, psrc_loss%.4f, psrc_acc%.4f, pte_loss%.4f, pte_acc%.4f" % \
            (epoch, train_loss, train_correct, test_loss, test_correct, ps_loss, ps_correct, pt_loss, pt_correct))


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
trainset, testset, full_patch_testset, source_trainset, source_testset = prepare_datasets(SOURCE_CLASS, TARGET_CLASS)
# pretrain or train from scratch
ckpt_dir = '/dockerdata/mavisbai/pretrain/ResNet18_{}.pth'.format(DATASET)
if os.path.exists(ckpt_dir):
    print("load pretrained model")
    model.load_state_dict(torch.load(ckpt_dir))
else:
    print("no pretrained model, train from scratch")
    if DATASET == 'ImageNet' or DATASET == 'TinyImageNet':
        ckpt_dir_ = '/dockerdata/mavisbai/pretrain/ResNet18_ImageNet_Download.pth'
        model = ResNet18_i(1000).to(device)
        model.load_state_dict(torch.load(ckpt_dir_))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, CLASS_NUM)
        model = model.to(device)
    train_model(model, trainset, testset, source_testset, full_patch_testset, init=True)
    torch.save(model.state_dict(), ckpt_dir)

# poisoned_dataset
poison_set, poison_ids, poison_lookup, poison_reverse_lookup = prepare_poisonset(model, trainset, TARGET_CLASS,
                                                                                 POISON_NUM)

# source_gradient
source_grad, source_grad_norm = get_gradient(model,
                                             torch.utils.data.DataLoader(source_trainset, batch_size=128, shuffle=False,
                                                                         drop_last=False), nn.CrossEntropyLoss())
poison_deltas = initialize_poison_deltas(POISON_NUM, trainset[0][0].shape, EPS, device)
att_optimizer = torch.optim.Adam([poison_deltas], lr=0.025, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[CRAFT_ITERS // 2.667,
                                                                            CRAFT_ITERS // 1.6,
                                                                            CRAFT_ITERS // 1.142],
                                                 gamma=0.1)
poison_deltas.grad = torch.zeros_like(poison_deltas)
# poison_bounds = torch.zeros_like(poison_deltas)
dataloader = torch.utils.data.DataLoader(poison_set, batch_size=128, drop_last=False, shuffle=False)
poison_tuples = []
# criterion = torch.nn.CrossEntropyLoss(size_average=False)
# for param in model.parameters():
#     param.requires_grad = True

for t in range(1, CRAFT_ITERS + 1):
    # print("===========iter %d==========="%t)
    base = 0
    target_losses, poison_correct = 0., 0.
    poison_imgs = []
    model.eval()
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        loss, prediction = batched_step(model, imgs, targets, poison_deltas, list(range(base, base + len(imgs))),
                                        F.cross_entropy, source_grad, source_grad_norm, device)
        target_losses += loss
        poison_correct += prediction
        base += len(imgs)
        poison_imgs.append(imgs)

    poison_deltas.grad.sign_()
    att_optimizer.step()
    scheduler.step()
    att_optimizer.zero_grad()
    with torch.no_grad():
        # Projection Step
        poison_imgs = torch.cat(poison_imgs)
        poison_deltas.data = torch.max(torch.min(poison_deltas, torch.ones_like(poison_deltas) * EPS),
                                       -torch.ones_like(poison_deltas) * EPS)
        poison_deltas.data = torch.max(torch.min(poison_deltas, 1 - poison_imgs), -poison_imgs)

    target_losses = target_losses / (len(dataloader) + 1)
    poison_acc = poison_correct / len(dataloader.dataset)
    if t % 10 == 0:
        print(f'Iteration {t}: Target loss is {target_losses:2.4f}, '
              f'Poison clean acc is {poison_acc * 100:2.2f}%')
    if t % RETRAIN_ITERS == 0:
        temp_poison_trainset = generate_poisoned_trainset(trainset, poison_deltas, poison_ids, poison_reverse_lookup)
        model = get_model()
        if DATASET == 'ImageNet' or DATASET == 'TinyImageNet':
            ckpt_dir_ = '/dockerdata/mavisbai/pretrain/ResNet18_ImageNet_Download.pth'
            model = ResNet18_i(1000).to(device)
            model.load_state_dict(torch.load(ckpt_dir_))
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, CLASS_NUM)
            model = model.to(device)
        train_model(model, temp_poison_trainset, testset, source_testset, full_patch_testset)
        source_grad, source_grad_norm = get_gradient(model, torch.utils.data.DataLoader(source_trainset, batch_size=128,
                                                                                        shuffle=False, drop_last=False),
                                                     nn.CrossEntropyLoss())

ckpt_dir_ = '/dockerdata/mavisbai/unsa/ResNet18_{}_dis_{}_{}_{}_{}_{}_{}.pth'.format(DATASET, BETA, POISON_NUM, PATCH_SIZE, CLASS_NUM, SOURCE_CLASS, TARGET_CLASS)
torch.save(model.state_dict(), ckpt_dir_)

