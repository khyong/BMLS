import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from utils import mixup_utils, bmls_utils

__all__ = [
    'FCCustom',
    'ETF',
    'WETF',
    'MultiWrapper',
]


def masked_linear(x, weight, bias, targets=None):
    if targets is None:
        return F.linear(x, weight, bias)
    else:
        output = torch.matmul(x, weight[targets,:].T)
        if bias is not None:
            output += bias[targets]
        return output


class FCCustom(nn.Linear):
    def __init__(self, *args, cfg=None, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, targets=None, **kwargs):
        return masked_linear(input, self.weight, self.bias, targets=targets)


class ETF(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        n_cls = num_classes

        
        self.BN_H = nn.BatchNorm1d(num_features)

        P = self.generate_random_orthogonal_matrix(num_features, n_cls)
        I = torch.eye(n_cls)
        one = torch.ones(n_cls, n_cls)
        M = np.sqrt(float(n_cls)/(n_cls-1)) * torch.matmul(P, I-((1./n_cls)*one))
        self.ori_M = M
        self.register_buffer('weight', M.T)
        self.register_buffer('bias', None)

    def generate_random_orthogonal_matrix(self, num_features, n_cls):
        a = np.random.random(size=(num_features, n_cls))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(n_cls), atol=1e-07), \
            torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(n_cls)))
        return P

    def forward(self, x, targets=None, **kwargs):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        if 'classifier' in kwargs:
            return masked_linear(x, self.weight, self.bias, targets=targets)
        else:
            return x


class WETF(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        n_cls = num_classes

        
        self.BN_H = nn.BatchNorm1d(num_features)

        P = self.generate_random_orthogonal_matrix(num_features, n_cls)
        I = torch.eye(n_cls)
        one = torch.ones(n_cls, n_cls)
        M = np.sqrt(float(n_cls)/(n_cls-1)) * torch.matmul(P, I-((1./n_cls)*one))
        self.ori_M = M
        self.register_buffer('weight', M.T)
        self.register_buffer('bias', None)

        self.w = nn.Parameter(torch.ones(num_classes, 1))
        self.b = nn.Parameter(torch.zeros(num_classes, 1))

    def generate_random_orthogonal_matrix(self, num_features, n_cls):
        a = np.random.random(size=(num_features, n_cls))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(n_cls), atol=1e-07), \
            torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(n_cls)))
        return P

    def forward(self, x, targets=None, **kwargs):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        if 'classifier' in kwargs:
            return masked_linear(x, self.weight, self.bias, targets=targets)
        else:
            return x

class MultiWrapper(nn.Module):
    def __init__(self, cfg, classifier):
        super().__init__()
        self.cfg = cfg
        self.clf_type = cfg.classifier.type
        self.clf = classifier
        self.lbl_mix2new = None
        self.lam_type = 'etf' if self.clf_type == 'ETF' else 'linear'

        self.mixed_bias = None

    def init_lbl_mix2new(self, lbl_mix2new):
        self.lbl_mix2new = np.stack([[v1, v2] for v1, v2 in lbl_mix2new.keys()])

    def init_mixed_weights(self, lam):
        v_lam_a, v_lam_b = mixup_utils.get_lam_pair(lam, type_=self.lam_type)

        weight = self.clf.w * self.clf.weight if self.clf_type == 'WETF' else self.clf.weight

        self.mixed_weight = v_lam_a * weight[self.lbl_mix2new[:,0]] + \
                            v_lam_b * weight[self.lbl_mix2new[:,1]]
        chk_same_lbl = np.where(self.lbl_mix2new[:,0]==self.lbl_mix2new[:,1])[0]
        if len(chk_same_lbl) != 0:
            self.mixed_weight[chk_same_lbl] = self.clf.weight[self.lbl_mix2new[chk_same_lbl][:,0]]

        if self.clf.bias is not None:
            self.mixed_bias = v_lam_a * self.clf.bias[self.lbl_mix2new[:,0]] + \
                              v_lam_b * self.clf.bias[self.lbl_mix2new[:,1]]

    def init_mixed_weights_batch(self, lam, tgts_a, tgts_b):
        v_lam_a, v_lam_b = mixup_utils.get_lam_pair(lam, type_=self.lam_type)

        pair_tgts = torch.hstack([tgts_a.reshape(-1, 1), tgts_b.reshape(-1, 1)])
        uni_pair_tgts, new_tgts = torch.unique(pair_tgts, dim=0, return_inverse=True)

        weight = self.clf.w * self.clf.weight if self.clf_type == 'WETF' else self.clf.weight

        self.mixed_weight = v_lam_a * weight[uni_pair_tgts[:,0]] + \
                            v_lam_b * weight[uni_pair_tgts[:,1]]
        chk_same_lbl = torch.where(uni_pair_tgts[:,0]==uni_pair_tgts[:,1])[0]
        if len(chk_same_lbl) != 0:
            self.mixed_weight[chk_same_lbl] = weight[tgts_a[chk_same_lbl]]

        if self.clf.bias is not None:
            self.mixed_bias = v_lam_a * self.clf.bias[uni_pair_tgts[:,0]] + \
                              v_lam_b * self.clf.bias[uni_pair_tgts[:,1]]

        return uni_pair_tgts, new_tgts

    def forward(self, x, train=False, targets=None):
        if train:
            if self.cfg.classifier.type in ['ETF', 'WETF']:
                x = self.clf(x)
            return masked_linear(
                x, self.mixed_weight, self.mixed_bias, targets=targets)
        else:
            if self.cfg.classifier.type in ['ETF', 'WETF']:
                return self.clf(x, classifier=True)
            else:
                return self.clf(x)

