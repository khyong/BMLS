import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from utils import etf_utils 

__all__ = [
    'CrossEntropyCustom',
    'DRLoss',
]


class CrossEntropyCustom(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super(CrossEntropyCustom, self).__init__()

    def forward(self, output, targets, **kwargs):
        return F.cross_entropy(output, targets, reduction='none')


class DRLoss(CrossEntropyCustom):
    def __init__(self, param_dict=None, **kwargs):
        super().__init__(param_dict=param_dict, **kwargs)

