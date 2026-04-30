import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import os
import sys
import ast
import click
import shutil
import random
import argparse
import warnings
import numpy as np
from itertools import combinations_with_replacement
from setproctitle import setproctitle

import _init_paths
import loss as custom_loss
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
    get_sampler,
)
from core.function import train_model, valid_model, test_model
from core.trainer import Trainer
from utils.reprod import fix_seed
from utils.dist import setup, cleanup


def parse_args():
    parser = argparse.ArgumentParser(description="Codes for bmls")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main_worker(rank, world_size, args):
    # ----- BEGIN basic setting -----
    update_config(cfg, args)
    logger = None

    verbose = (not cfg.ddp) or (rank == 0)
    if verbose:
        logger, log_file = create_logger(cfg)
        warnings.filterwarnings("ignore")

    fix_seed(cfg.seed_num)

    if cfg.ddp:
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size, port=cfg.port)

    torch.cuda.set_device(rank)
    # ----- END basic setting -----

    # ----- BEGIN dataset setting -----
    transform_tr = get_transform(cfg, mode='train')
    transform_ts = get_transform(cfg, mode='test')

    train_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=True, download=True, transform=transform_tr)

    if not isinstance(train_set.targets, torch.Tensor):
        train_set.targets = torch.tensor(train_set.targets, dtype=torch.long)
    num_classes = len(torch.unique(train_set.targets))
    num_class_list, ctgy_list = get_category_list(train_set.targets, num_classes, cfg)

    param_dict = {
        'num_classes': num_classes,
        'num_class_list': num_class_list,
        'cfg': cfg,
        'rank': rank,
    }

    class_map = train_set.class_map if cfg.dataset.dataset in ['ImageNetLT', 'PlacesLT', 'iNa2018'] else None
    valid_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=False, download=True, transform=transform_ts, class_map=class_map)
    
    # get sampler
    trainsampler = get_sampler(cfg, train_set, param_dict=param_dict)
    if cfg.train.sampler.type == 'bmls':
        train_set = custom_dataset.MixedLabelDataset(train_set)
    validsampler = DistributedSampler(valid_set) if cfg.ddp else None
    
    if cfg.ddp:
        batch_size = int(cfg.train.batch_size / world_size)
        num_workers = int((cfg.train.num_workers+world_size-1)/world_size)
    else:
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers

    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(trainsampler is None),
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        sampler=trainsampler,
    )

    validloader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        sampler=validsampler,
    )
    # ----- END dataset setting -----

    # ----- BEGIN model builder -----
    num_epochs = cfg.train.num_epochs

    model = get_model(cfg, num_classes, rank)
    if cfg.pretrained: # load pretrained model
        if os.path.isfile(cfg.pretrained):
            print("=> loading checkpoint '{}'".format(cfg.pretrained))
            checkpoint = torch.load(cfg.pretrained, map_location='cuda:{}'.format(rank))
            model.load_state_dict(checkpoint['state_dict'])
    mm = model.module if cfg.ddp or cfg.dp else model
    trainer = Trainer(cfg, rank)
    criterion = getattr(custom_loss, cfg.loss.loss_type)(param_dict=param_dict).cuda(rank)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- END model builder -----

    # ----- BEGIN recording setting -----
    if verbose:
        model_dir = os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "models")
        code_dir = os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "codes")
        tensorboard_dir = (
            os.path.join(cfg.output_dir, cfg.name, 'seed{:03d}'.format(cfg.seed_num), "tensorboard")
            if cfg.train.tensorboard.enable else None
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            shutil.rmtree(code_dir)
            if (tensorboard_dir is not None) and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        current_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            '*.pyc', '*.so', '*.out', '*pycache*', '*.pth', '*build*', '*output*', '*datasets*'
        )
        shutil.copytree(os.path.join(current_dir, '..'), code_dir, ignore=ignore)

        if tensorboard_dir is not None:
            dummy_input = torch.rand((1, 3) + cfg.input_size).cuda(rank)
            writer = SummaryWriter(log_dir=tensorboard_dir)
            pooling_module = mm.pooling
            writer.add_graph(pooling_module, (dummy_input,))
        else:
            writer = None
    # ----- END recording setting -----

    # ----- START train & valid -----
    best_result, best_epoch, start_epoch = 0, 0, 1
    save_step = cfg.save_step if cfg.save_step != -1 else num_epochs

    if verbose:
        logger.info(
            "-------------------Train start: {} {} {} | {} {} | {} / {}--------------------".format(
                cfg.backbone.type, cfg.pooling.type, cfg.reshape.type, 
                cfg.classifier.type, cfg.scaling.type, 
                cfg.loss.loss_type,
                cfg.train.trainer.type,
            )
        )

    kwargs_tr, kwargs_val = {}, {}
    # for Imbalanced Learning
    if cfg.dataset.type == 'imbalanced':
        kwargs_tr['lt'], kwargs_val['lt'] = True, True
        if cfg.train.sampler.type == 'cas':
            kwargs_tr['num_batches'] = int(np.ceil(float(len(train_set))/cfg.train.batch_size))
    if cfg.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        kwargs_tr['scaler'] = scaler

    # OURS - start: init attributes for bmls
    if cfg.train.sampler.type == 'bmls':
        palette, remainder, lbl_mix2new = trainsampler.get_palette()
        if remainder == 0:
            trainsampler.set_palette_cp()
        kwargs_tr['lbl_mix2new'] = lbl_mix2new
    if cfg.train.trainer.type.endswith('multi'):
        if cfg.train.sampler.type != 'bmls' or lbl_mix2new is None:
            seq = [cls_num for cls_num in range(num_classes)]
            lbl_mix2new = {
                tuple(v): i for i, v in enumerate(combinations_with_replacement(seq, 2))}
            kwargs_tr['lbl_mix2new'] = lbl_mix2new
        mm.classifier.init_lbl_mix2new(lbl_mix2new)
    if cfg.train.sampler.pair_type in ['bbml', 'gbml']:
        kwargs_tr['cnt_map'] = np.zeros((num_classes, num_classes), dtype=np.uint8)
    # OURS - end: init attributes for bmls

    for epoch in range(start_epoch, num_epochs + 1):
        if (epoch > start_epoch) and (scheduler is not None):
            scheduler.step()
        if cfg.ddp:
            trainsampler.set_epoch(epoch)

        # OURS - start: sampling for balancely mixed samples
        if cfg.train.sampler.type == 'bmls' or cfg.train.sampler.pair_type in ['bbml', 'gbml']:
            mixup_alpha = cfg.train.trainer.mixup_alpha
            kwargs_tr['mixup_lam'] = np.random.beta(mixup_alpha, mixup_alpha) \
                if mixup_alpha > 0 else 1

        if cfg.train.sampler.type == 'bmls':
            if remainder > 0:
                trainsampler.distribute_remainder(palette, remainder)

            logger.info("num_labels: {}, total_num_samples: {}".format(
                len(lbl_mix2new), np.sum(trainsampler.palette_cp)))
            logger.info(trainsampler.palette_cp)
        # OURS - end: sampling for balancely mixed samples

        # train
        train_acc, train_loss = train_model(
            trainloader, model, epoch, num_epochs, optimizer, trainer, 
            criterion, cfg, logger, verbose, **kwargs_tr
        )

        loss_dict, acc_dict = {'train_loss': train_loss}, {'train_acc': train_acc}

        # valid
        if (cfg.valid_step != -1) and (epoch % cfg.valid_step == 0):
            valid_acc, valid_loss, lt_acc = valid_model(
                validloader, model, epoch, 
                criterion, cfg, logger, verbose, rank, **kwargs_val
            )
            loss_dict['valid_loss'], acc_dict['valid_acc'] = valid_loss, valid_acc
            if verbose:
                save_paths = []
                # save current model
                if epoch % save_step == 0:
                    cur_path = os.path.join(model_dir, 'epoch_{}.pth'.format(epoch))
                    save_paths.append(cur_path)
                # save best model
                if valid_acc >= best_result:
                    best_epoch, best_result, best_lt = epoch, valid_acc, lt_acc
                    best_path = os.path.join(model_dir, 'best_model.pth')
                    save_paths.append(best_path)

                if len(save_paths) > 0:
                    dump_result_and_model(
                        save_paths, epoch, valid_acc,
                        model, optimizer, scheduler,
                        lt_result=lt_acc, save_only_result=cfg.save_only_result)

                pbar_str = "------- Best : Epoch:{:>3d}".format(best_epoch) \
                    + "               " \
                    + " val_acc:{:>5.2f}%".format(best_result * 100) \
                    + " | many:{:>5.2f}%".format(best_lt[0] * 100) \
                    + " | med :{:>5.2f}%".format(best_lt[1] * 100) \
                    + " | few :{:>5.2f}%".format(best_lt[2] * 100)
                logger.info(pbar_str)

        if cfg.train.tensorboard.enable and verbose:
            writer.add_scalars('scalar/acc', acc_dict, epoch)
            writer.add_scalars('scalar/loss', loss_dict, epoch)
    if cfg.train.tensorboard.enable and verbose:
        writer.close()
    if verbose:
        logger.info(
            "-------------------Train Finished: {} (seed:{})-------------------".format(cfg.name, cfg.seed_num)
        )
        if not cfg.ddp and not cfg.save_only_result:
            test_model(
                validloader, cfg, rank, verbose,
                num_classes=num_classes, pretrained=best_path
            )

    if cfg.ddp:
        cleanup()
    # ----- END train & valid -----

def dump_result_and_model(
    save_paths, epoch, result, model, optimizer, scheduler,
    lt_result=None, save_only_result=False,
):

    save_dict = {
        'epoch': epoch,
        'result': result,
    }
    if lt_result is not None:
        save_dict['lt_result'] = lt_result
    if not save_only_result:
        save_dict['state_dict'] = model.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        save_dict['scheduler'] = scheduler.state_dict() if scheduler is not None else None

    for save_path in save_paths:
        torch.save(save_dict, save_path)
        

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    setproctitle(cfg.name)

    if cfg.ddp:
        ngpus_per_node = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        rank = cfg.rank if cfg.rank != -1 else 0
        main_worker(rank, cfg.world_size, args)

