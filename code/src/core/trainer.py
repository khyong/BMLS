import torch
import torch.nn.functional as F

import math
import numpy as np
from copy import deepcopy
from collections import Counter
from contextlib import nullcontext

import utils.mixup_utils as mixup_utils
import utils.etf_utils as etf_utils
import utils.bmls_utils as bmls_utils
from core.evaluate import accuracy


class Trainer:
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.trainer_type = cfg.train.trainer.type
        self.rank = rank
        self.num_epochs = cfg.train.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.init_all_params()

    def init_all_params(self):
        self.batch_wise = self.cfg.loss.batch_wise
        self.pair_type = self.cfg.train.sampler.pair_type
        self.mixup_alpha = self.cfg.train.trainer.mixup_alpha
    
    def reset_epoch(self, epoch):
        self.epoch = epoch

    def _with_autocast(self):
        return torch.cuda.amp.autocast() if self.cfg.mixed_precision else nullcontext()

    def _with_freeze(self):
        return torch.no_grad() if self.cfg.backbone.backbone_freeze else nullcontext()

    def forward(self, model, criterion, data, targets, **kwargs):
        return getattr(Trainer, self.trainer_type)(
            self, model, criterion, data, targets, **kwargs
        )

    def default(self, model, criterion, data, targets, **kwargs):
        """
            `sampler`: default, cbs, cas
        """
        data, targets = data.cuda(self.rank), targets.cuda(self.rank)

        if self.batch_wise:
            uni_tgts, pred_tgts = torch.unique(targets, return_inverse=True)
            kwargs_clf = {'targets': uni_tgts}
        else:
            kwargs_clf, pred_tgts = {}, targets

        # encode
        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            # classify
            output = model(features, classifier_flag=True, **kwargs_clf)
            # loss
            loss = criterion(output, pred_tgts).mean()
        
        # prediction
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), pred_tgts.cpu().numpy())[0]

        if 'visual' in kwargs:
            color_cls_chk = bmls_utils.get_color(
                targets, targets, 1., self.num_classes,
                rank=self.rank, class_check=True)
            h = features.clone().detach().cpu()
            kwargs['visual']['features'].append(h)
            kwargs['visual']['y'].append(
                np.hstack([
                    targets.detach().cpu().numpy()[:,None],
                    targets.detach().cpu().numpy()[:,None]
                ])
            )
            kwargs['visual']['colors_class'].append(color_cls_chk)

        return loss, acc

    def mixup(self, model, criterion, data, targets, **kwargs):
        """
            `sampler`: default, cbs, cas, bmls
        """
        cnt_map = None if 'cnt_map' not in kwargs else kwargs['cnt_map']
        data_a, data_b, tgts_a, tgts_b = mixup_utils.pair_data(
            data, targets, pair_type=self.pair_type, cnt_map=cnt_map)
        
        lam = None if 'mixup_lam' not in kwargs else kwargs['mixup_lam']
        mixed_data, lam = mixup_utils.mixup_data(
            data_a, data_b, alpha=self.mixup_alpha, lam=lam)

        mixed_data = mixed_data.cuda(self.rank)
        tgts_a, tgts_b = tgts_a.cuda(self.rank), tgts_b.cuda(self.rank)

        if self.batch_wise:
            pair_tgts = torch.hstack([tgts_a.reshape(-1, 1), tgts_b.reshape(-1, 1)])
            uni_tgts, src_tgts = torch.unique(pair_tgts, return_inverse=True)
            kwargs_clf = {'targets': uni_tgts}
            src_tgts_a, src_tgts_b = src_tgts[:,0], src_tgts[:,1]
        else:
            kwargs_clf = {}
            src_tgts_a, src_tgts_b = tgts_a, tgts_b
        pred_tgts = mixup_utils.get_pred_targets(src_tgts_a, src_tgts_b, lam)

        # encode
        with self._with_autocast():
            with self._with_freeze():
                features = model(mixed_data, feature_flag=True)
            # classify
            output = model(features, classifier_flag=True, **kwargs_clf)
            # loss
            loss = mixup_utils.mixup_criterion(
                criterion, output, src_tgts_a, src_tgts_b, lam).mean()

        # prediction
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), pred_tgts.cpu().numpy())[0]

        if 'visual' in kwargs:
            color_cls_chk = bmls_utils.get_color(
                tgts_a, tgts_b, lam, self.num_classes,
                rank=self.rank, class_check=True)
            h = features.clone().detach().cpu()
            kwargs['visual']['features'].append(h)
            kwargs['visual']['y'].append(
                np.hstack([
                    tgts_a.detach().cpu().numpy()[:,None],
                    tgts_b.detach().cpu().numpy()[:,None]
                ])
            )
            kwargs['visual']['colors_class'].append(color_cls_chk)

        return loss, acc

    def multi(self, model, criterion, data, targets, **kwargs):
        """
            `sampler`: default, cbs, cas, bmls
        """
        cnt_map = None if 'cnt_map' not in kwargs else kwargs['cnt_map']
        data_a, data_b, tgts_a, tgts_b = mixup_utils.pair_data(
            data, targets, pair_type=self.pair_type, cnt_map=cnt_map)
        
        lam = None if 'mixup_lam' not in kwargs else kwargs['mixup_lam']
        mixed_data, lam = mixup_utils.mixup_data(
            data_a, data_b, alpha=self.mixup_alpha, lam=lam)

        mixed_data = mixed_data.cuda(self.rank)
        tgts_a, tgts_b = tgts_a.cuda(self.rank), tgts_b.cuda(self.rank)

        mm_clf = model.module.classifier if self.cfg.ddp else model.classifier
        if self.batch_wise:
            uni_pair_tgts, pred_tgts = mm_clf.init_mixed_weights_batch(lam, tgts_a, tgts_b)
        else:
            mm_clf.init_mixed_weights(lam)
            pred_tgts = bmls_utils.convert_to_singleton(
                tgts_a, tgts_b, kwargs['lbl_mix2new'], rank=self.rank)

        # encode
        with self._with_autocast():
            with self._with_freeze():
                features = model(mixed_data, feature_flag=True)
            # classify
            output = model(features, classifier_flag=True, train=True)
            # loss
            loss = criterion(output, pred_tgts).mean()

        # prediction
        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), pred_tgts.cpu().numpy())[0]

        if 'visual' in kwargs:
            color_cls_chk = bmls_utils.get_color(
                tgts_a, tgts_b, lam, self.num_classes,
                rank=self.rank, class_check=True)
            h = features.clone().detach().cpu()
            kwargs['visual']['features'].append(h)
            kwargs['visual']['y'].append(
                np.hstack([
                    tgts_a.detach().cpu().numpy()[:,None],
                    tgts_b.detach().cpu().numpy()[:,None]
                ])
            )
            kwargs['visual']['colors_class'].append(color_cls_chk)

        return loss, acc

    def etf_mixup(self, model, criterion, data, targets, **kwargs):
        """
            `sampler`: default, cbs, cas, bmls
        """
        cnt_map = None if 'cnt_map' not in kwargs else kwargs['cnt_map']
        data_a, data_b, tgts_a, tgts_b = mixup_utils.pair_data(
            data, targets, pair_type=self.pair_type, cnt_map=cnt_map)
        
        lam = None if 'mixup_lam' not in kwargs else kwargs['mixup_lam']
        mixed_data, lam = mixup_utils.mixup_data(
            data_a, data_b, alpha=self.mixup_alpha, lam=lam)

        mixed_data = mixed_data.cuda(self.rank)
        tgts_a, tgts_b = tgts_a.cuda(self.rank), tgts_b.cuda(self.rank)

        # init class weights
        mm_clf = model.module.classifier if self.cfg.ddp else model.classifier
        ori_M = mm_clf.ori_M
        learned_norm = etf_utils.produce_Ew(tgts_a, self.num_classes)
        cur_M = learned_norm * ori_M.cuda(self.rank)

        # encode
        with self._with_autocast():
            with self._with_freeze():
                features = model(mixed_data, feature_flag=True)
                features = mm_clf(features)
            with torch.no_grad(): 
                feat_nograd = features.detach()
                H_length = torch.clamp(
                    torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
            if self.cfg.loss.loss_type == 'DRLoss':
                # classify and calculate loss
                loss_a = etf_utils.dot_loss(
                    features, tgts_a, cur_M, H_length, 
                    reg_lam=0. if 'CIFAR' in self.cfg.dataset.dataset else 0.01,
                    type_='reg_dot_loss')
                loss_b = etf_utils.dot_loss(
                    features, tgts_b, cur_M, H_length, 
                    reg_lam=0. if 'CIFAR' in self.cfg.dataset.dataset else 0.01,
                    type_='reg_dot_loss')
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # loss
                loss = mixup_utils.mixup_criterion(
                    criterion, output, tgts_a, tgts_b, lam).mean()

        # prediction
        with torch.no_grad():
            output = torch.matmul(features, cur_M)
        pred_tgts = mixup_utils.get_pred_targets(tgts_a, tgts_b, lam)

        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), pred_tgts.cpu().numpy())[0]

        return loss, acc

    def etf_multi(self, model, criterion, data, targets, **kwargs):
        """
            `sampler`: default, cbs, cas, bmls
        """
        cnt_map = None if 'cnt_map' not in kwargs else kwargs['cnt_map']
        data_a, data_b, tgts_a, tgts_b = mixup_utils.pair_data(
            data, targets, pair_type=self.pair_type, cnt_map=cnt_map)
        
        lam = None if 'mixup_lam' not in kwargs else kwargs['mixup_lam']
        mixed_data, lam = mixup_utils.mixup_data(
            data_a, data_b, alpha=self.mixup_alpha, lam=lam)

        mixed_data = mixed_data.cuda(self.rank)
        tgts_a, tgts_b = tgts_a.cuda(self.rank), tgts_b.cuda(self.rank)

        mm_clf = model.module.classifier if self.cfg.ddp else model.classifier
        if self.batch_wise:
            uni_pair_tgts, pred_tgts = mm_clf.init_mixed_weights_batch(lam, tgts_a, tgts_b)
        else:
            if self.cfg.train.sampler.type != 'bmls':
                mm_clf.init_mixed_weights(lam)
            pred_tgts = bmls_utils.convert_to_singleton(
                tgts_a, tgts_b, kwargs['lbl_mix2new'], rank=self.rank)

        # init class weights
        ori_M = mm_clf.mixed_weight.T
        cur_M = ori_M.cuda(self.rank)
        #learned_norm = etf_utils.produce_Ew(pred_tgts, len(uni_pair_tgts))
        #cur_M = learned_norm * ori_M.cuda(self.rank)

        # encode
        with self._with_autocast():
            with self._with_freeze():
                features = model(mixed_data, feature_flag=True)
            #if self.cfg.classifier.type == 'ETF':
            #    features = mm_clf.clf(features)
            #with torch.no_grad(): 
            #    feat_nograd = features.detach()
            #    H_length = torch.clamp(
            #        torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
            if self.cfg.loss.loss_type == 'DRLoss':
                # classify and calculate loss
                loss = etf_utils.dot_loss(
                    features, pred_tgts, cur_M, H_length, 
                    reg_lam=0. if 'CIFAR' in self.cfg.dataset.dataset else 0.01,
                    type_='reg_dot_loss')
            else:
                # classify
                output = torch.matmul(features, cur_M)
                # loss
                loss = criterion(output, pred_tgts).mean()

        # prediction
        with torch.no_grad():
            output = torch.matmul(features, cur_M)

        pred = torch.argmax(output, 1)
        acc = accuracy(pred.cpu().numpy(), pred_tgts.cpu().numpy())[0]

        return loss, acc

