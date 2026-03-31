from .basic import CustomMNIST as MNIST
from .basic import CustomFMNIST as FashionMNIST
from .basic import CustomCIFAR10 as CIFAR10
from .basic import CustomCIFAR100 as CIFAR100
from .basic import ImageNet1k
from .basic import CustomMixedDataset
from .colored_mnist import ColoredMNIST
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from .ina2018 import iNa2018
from .imbalance_imagenet import ImageNetLT
from .imbalance_places import PlacesLT
from .sampler import *

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MixedLabelDataset(Dataset):
    def __init__(self, dataset):
        super(MixedLabelDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        img_a, targets_a = self.dataset.__getitem__(index[0])
        img_b, targets_b = self.dataset.__getitem__(index[1])
        return (img_a, img_b), (targets_a, targets_b)

    def __len__(self):
        return len(self.dataset)

