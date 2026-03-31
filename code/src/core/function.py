import torch
import torch.nn.functional as F

import os
import time
import numpy as np

import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
from utils.utils import get_model
from utils import etf_utils, bmls_utils


def train_model(
    trainloader, model, epoch, num_epochs, optimizer, trainer, 
    criterion, cfg, logger, verbose, **kwargs
):
    if cfg.eval_mode:
        model.eval()
    else:
        if cfg.backbone.backbone_freeze:
            model.backbone.eval()
            model.pooling.eval()
            model.reshape.eval()
            model.classifier.train()
            model.scaling.train()
        else:
            model.train()

    start_time = time.time()

    num_batches = len(trainloader) if 'num_batches' not in kwargs else kwargs['num_batches']
    scaler = None if 'scaler' not in kwargs else kwargs['scaler']
    warmup_to = None if 'warmup_to' not in kwargs else kwargs['warmup_to']

    tr_loss = AverageMeter()
    tr_acc = AverageMeter()

    trainer.reset_epoch(epoch)
    for i, (data, targets) in enumerate(trainloader):
        if i > num_batches:
            break

        cnt = targets.shape[0] if not isinstance(targets, list) else targets[0].shape[0]
        loss, acc = trainer.forward(model, criterion, data, targets, **kwargs)

        tr_loss.update(loss.data.item(), cnt)
        tr_acc.update(acc, cnt)

        if cfg.mixed_precision:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i & cfg.show_step == 0) and verbose:
            pbar_str = "Epoch:{:>3d} [{:>3d}/{}] loss:{:>5.3f} acc:{:>5.2f}%".format(
                epoch, i, num_batches, tr_loss.val, tr_acc.val * 100)
            logger.info(pbar_str)
    end_time = time.time()
    if verbose:
        pbar_str = "---Epoch:{:>3d}/{}".format(epoch, num_epochs) \
            + " tr_loss:{:>5.3f}".format(tr_loss.avg) \
            + " tr_acc:{:>5.2f}%".format(tr_acc.avg * 100) \
            + " elapsed_time:{:>5.2f}m---".format((end_time - start_time)/60)
        logger.info(pbar_str)

    return tr_acc.avg, tr_loss.avg


def valid_model(
    dataloader, model, epoch,
    criterion, cfg, logger, verbose, rank, **kwargs
):
    model.eval()
    fusion_matrix = FusionMatrix(cfg.dataset.num_classes)
    with torch.no_grad():
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.cuda(rank), targets.cuda(rank)

            features = model(data, feature_flag=True, **kwargs)
            if cfg.classifier.type in ['ETF', 'WETF']:
                # init constants
                mm_clf = model.module.classifier if cfg.ddp else model.classifier
                if cfg.classifier.type == 'WETF':
                    if cfg.train.trainer.type.endswith('multi'):
                        cur_M = mm_clf.clf.w.T * mm_clf.clf.ori_M.cuda(rank)
                        #features = mm_clf.clf(features)
                    else:
                        cur_M = mm_clf.w.T * mm_clf.ori_M.cuda(rank)
                        #features = mm_clf(features)
                else:
                    if cfg.train.trainer.type.endswith('multi'):
                        cur_M = mm_clf.clf.ori_M.cuda(rank)
                        #features = mm_clf.clf(features)
                    else:
                        cur_M = mm_clf.ori_M.cuda(rank)
                        #features = mm_clf(features)

                #feat_nograd = features.detach()
                #H_length = torch.clamp(
                #    torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)

                output = torch.matmul(features, cur_M)
                if cfg.loss.loss_type == 'CrossEntropyCustom':
                    loss = criterion(output, targets).mean()
                else:
                    loss = etf_utils.dot_loss(
                        features, targets, cur_M, H_length, 
                        reg_lam=0. if 'CIFAR' in cfg.dataset.dataset else 0.01,
                        type_='reg_dot_loss')
            else:
                output = model(features, classifier_flag=True, **kwargs)
                loss = criterion(output, targets).mean()

            val_loss.update(loss.data.item(), targets.shape[0])

            pred = torch.argmax(output, 1)
            acc, cnt = accuracy(pred.cpu().numpy(), targets.cpu().numpy())
            val_acc.update(acc, cnt)

            fusion_matrix.update(pred.cpu().numpy(), targets.cpu().numpy())

            if 'visual' in kwargs:
                color_cls_chk = bmls_utils.get_color(
                    targets, targets, 1.0, cfg.dataset.num_classes,
                    rank=rank, class_check=True)
                h = features.clone().detach().cpu()
                kwargs['visual']['features'].append(h)
                kwargs['visual']['y'].append(
                    np.hstack([
                        targets.detach().cpu().numpy()[:,None],
                        targets.detach().cpu().numpy()[:,None]
                    ])
                )
                kwargs['visual']['colors_class'].append(color_cls_chk)

        if cfg.ddp:
            val_loss.all_reduce()
            val_acc.all_reduce()
            fusion_matrix.all_reduce()

        if 'lt' in kwargs:
            acc_classes = fusion_matrix.get_rec_per_class()
            acc_many = acc_classes[cfg.dataset.class_index.many[0]:cfg.dataset.class_index.many[1]].mean()
            acc_med = acc_classes[cfg.dataset.class_index.med[0]:cfg.dataset.class_index.med[1]].mean()
            acc_few = acc_classes[cfg.dataset.class_index.few[0]:cfg.dataset.class_index.few[1]].mean()

            if verbose:
                pbar_str = "------- Valid: Epoch:{:>3d}".format(epoch) \
                    + " val_loss:{:>5.3f}".format(val_loss.avg) \
                    + " val_acc:{:>5.2f}%".format(val_acc.avg * 100) \
                    + " | many:{:>5.2f}%".format(acc_many * 100) \
                    + " | med :{:>5.2f}%".format(acc_med * 100) \
                    + " | few :{:>5.2f}%".format(acc_few * 100)
                logger.info(pbar_str)

            return val_acc.avg, val_loss.avg, [acc_many, acc_med, acc_few]

        else:
            if verbose:
                pbar_str = "------Valid: Epoch:{:>3d}".format(epoch) \
                    + " val_loss:{:>5.3f}".format(val_loss.avg) \
                    + " val_acc:{:>5.2f}%".format(val_acc.avg * 100)
                logger.info(pbar_str)

            return val_acc.avg, val_loss.avg, None


def test_model(
    dataloader, cfg, rank, verbose,
    num_classes=10, pretrained=None
):
    model = get_model(cfg, num_classes, rank)
    
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location='cuda:{}'.format(rank))
        if cfg.ddp:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            ckpt_state_dict = dict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    ckpt_state_dict[k[7:]] = v
                else:
                    ckpt_state_dict[k] = v
            model.load_state_dict(ckpt_state_dict)
    model.eval()

    with torch.no_grad():
        ts_acc = AverageMeter()

        for i, (data, targets) in enumerate(dataloader):
            data = data.cuda(rank)

            output = model(data)
            pred = torch.argmax(output, 1)

            acc, cnt = accuracy(pred.cpu().numpy(), targets.numpy())
            ts_acc.update(acc, cnt)

        if cfg.ddp:
            ts_acc.all_reduce()

    if verbose:
        print("*** Test Accuracy: {:>5.2f}%".format(ts_acc.avg * 100))

