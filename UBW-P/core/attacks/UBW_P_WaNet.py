'''
This is the implement of UBW with WaNet-type trigger [1].

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR, 2021.
'''

import copy
from copy import deepcopy
import random

import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.transforms import Compose

from .base import *


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = self.grid + ins / self.h
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        return poison_img



class AddDatasetFolderTrigger(AddTrigger):
    """Add WaNet trigger to DatasetFolder images.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddDatasetFolderTrigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale




    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).
        Returns:
            torch.Tensor: The poisoned image.
        """
        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float)
            img = self.add_trigger(img, noise=self.noise)
            # 1 x H x W
            if img.size(0) == 1:
                img = img.squeeze().numpy()
                img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = img.numpy().transpose(1, 2, 0)
                img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.numpy()

            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0).numpy()

            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
            # H x W x C
            else:
                img = F.convert_image_dtype(img, torch.float)
                img = img.permute(2, 0, 1)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))



class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = self.add_trigger(img, noise=self.noise)
        img = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


'''
class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target
'''

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 identity_grid, 
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 num_class):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Get random labels
        self.random_label = np.random.randint(0, num_class, size=total_num)

        # add noise 
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid,  noise=True))

        '''
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        '''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            #print('before:', target)
            target = self.random_label[index] #self.poisoned_target_transform(target)
            #print('after:', target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            sample = self.poisoned_transform_noise(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # target = self.poisoned_target_transform(target)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedDatasetFolder_(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder_, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # add noise
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num + noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([])  # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform)  # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index,
                                                  AddDatasetFolderTrigger(identity_grid, noise_grid, noise=False))
        # add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index,
                                                        AddDatasetFolderTrigger(identity_grid, noise_grid, noise=True))

        '''
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        '''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            sample = self.poisoned_transform_noise(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # target = self.poisoned_target_transform(target)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 identity_grid, 
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 num_class):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Get random labels
        self.random_label = np.random.randint(0, num_class, size=total_num)

        # add noise 
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=True))

        '''
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        '''

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            #print('before:', target)
            target = self.random_label[index] #self.poisoned_target_transform(target)
            #print('after:', target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10_(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR10_, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # add noise
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=True))

        '''
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        '''

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index, num_class):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index, num_class)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index, num_class)
    else:
        raise NotImplementedError

def CreatePoisonedDataset_(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder_(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10_(benign_dataset, poisoned_rate, identity_grid, noise_grid, noise, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError



class UBW_P_WaNet(Base):
    """Construct poisoned datasets with UBW-P under WaNet-type trigger patterns.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        poisoned_rate (float): Ratio of poisoned samples.
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        num_class: number of classes of the dataset. Default: 10.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 num_class=10,
                 deterministic=False):
       
        super(UBW_P_WaNet, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            num_class)

        self.poisoned_test_dataset = CreatePoisonedDataset_(
            test_dataset,
            1.0,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_test_index,
            poisoned_target_transform_index)
