###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import model
# from feeder.feeder_density import Feeder
from feeder.feeder_train import Feeder
from utils import to_numpy
from utils.logging import Logger 
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint
from sklearn.metrics import precision_score, recall_score
from apex import amp
from utils.losses import normal, ohem, ohem3, class_balance_ce, focal_loss, FocalLoss1, FocalLoss2, weighted_ce
# from torch.utils.tensorboard import SummaryWriter

## for the shared memory
#from torch.utils.data import dataloader
#from multiprocessing.reduction import ForkingPickler
#default_collate_func = dataloader.default_collate
#def default_collate_override(batch):
#    dataloader._use_shared_memory = False
#    return default_collate_func(batch)
#setattr(dataloader, 'default_collate', default_collate_override)
#for t in torch._storage_classes:
#    if sys.version_info[0] == 2:
#        if t in ForkingPickler.dispatch:
#            del ForkingPickler.dispatch[t]
#    else:
#        if t in ForkingPickler._extra_reducers:
#            del ForkingPickler._extra_reducers[t]


def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--print_freq', default=20, type=int)

    # Optimization args
    parser.add_argument('--fp16', action='store_true', default=False, help='fp16 for train')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_adjust_list', type=int, nargs='+', default=[1,2,3,4])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--is_density', type=int, default=0)
    parser.add_argument('--clique_size', type=int, default=0)
    parser.add_argument('--rand_val', type=float, default=0)
    parser.add_argument('--rand_twice', type=int, default=0)
    parser.add_argument('--resample', type=float, default=0)

    # Training args
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/workspace/data/hgy_17w/hgyA_feat.lmdb'))
    parser.add_argument('--knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/workspace/data/hgy_17w/hgyA_feat/faiss_gpu_k_160.npy'))
    parser.add_argument('--density_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/mnt/disk/face_cluster/IJB/512/density.npy'))
    parser.add_argument('--label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/CASIA.labels.npy'))
    parser.add_argument('--local_label_path', type=str, default='/workspace/data/hgy_17w/pseudo_hgyAv2_0.20.npy')
    parser.add_argument('--mode', type=str, default='sup')
    parser.add_argument('--lambda_u', type=int, default=10)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[80,5])
    parser.add_argument('--active_connection', type=int, default=5)
    parser.add_argument('--loss', type=str, default='normal')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--model_drop', type=str, default='gcn')
    parser.add_argument('--block_size', type=int, default=3)
    parser.add_argument('--drop_prob', type=float, default=0.3)
    parser.add_argument('--density_th', type=float, default=0.6)
  
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    node_world_size = int(os.environ['WORLD_SIZE']) 
    ngpus_per_node = torch.cuda.device_count()

    logging.info("=> node_world_size [{}], gpus_per_node [{}]".format(
        node_world_size, ngpus_per_node
    ))
    world_size = ngpus_per_node * node_world_size
    logging.info ("=> total world size [{}]".format(world_size))


    #local_label_path_list = ['data/hgyA_v2/local_label/pseudo_hgyAv2_0.20.modify.npy', 'data/hgyA_v2/local_label/pseudo_hgyAv2_0.24.modify.npy', 'data/hgyA_v2/local_label/pseudo_hgyAv2_0.30.modify.npy']
    #logs_dir_list = [os.path.join(args.logs_dir, 'thres0.20'), os.path.join(args.logs_dir, 'thres0.24'), os.path.join(args.logs_dir, 'thres0.30')]
    #for i in range(len(local_label_path_list)):
    args.local_label_path = args.local_label_path
    os.makedirs(args.logs_dir, exist_ok=True)
    print("log dir:  ", args.logs_dir)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,  world_size, args))


def main_worker(gpu, ngpus_per_node, world_size, args):
    args.gpu = gpu
    torch.cuda.set_device(gpu)
    node_rank = int(os.environ['RANK'])
    process_rank = node_rank * ngpus_per_node + gpu

    dist.init_process_group(backend="nccl", world_size=world_size, rank=process_rank)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainset = Feeder(args.feat_path,
                      args.knn_graph_path,
                      args.label_path,
                      args.local_label_path,
                      args.density_path,
                      args.mode,
                      args.lambda_u,
                      args.seed,
                      args.k_at_hop,
                      args.active_connection,
                      args.rand_val,
                      args.rand_twice,
                      args.resample,
                      args.is_density,
                      args.clique_size,
                      args.loss)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
            trainset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.workers, shuffle=False, pin_memory=True)
    if args.model_drop == 'gcn':
        net = model.gcn(args.features).cuda(gpu)
    elif args.model_drop == 'gcn_drop_1':
        net = model.gcn_drop_1(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_2':
        net = model.gcn_drop_2(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_3':
        net = model.gcn_drop_3(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_4':
        net = model.gcn_drop_4(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_f1':
        net = model.gcn_drop_f1(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_f1_1':
        net = model.gcn_drop_f1_1(args.features, args.block_size, args.drop_prob).cuda(gpu)
    elif args.model_drop == 'gcn_drop_f1_1_self':
        net = model.gcn_drop_f1_1_self(args.features, args.block_size, args.drop_prob).cuda(gpu)

    args.lr = args.lr * world_size
    opt = torch.optim.SGD(net.parameters(), args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay) 

    if args.fp16:
        net, opt = amp.initialize(net, opt, opt_level="O1", loss_scale='dynamic')
    net = DDP(net, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(gpu)
    criterion_nll = nn.NLLLoss().cuda(gpu)
    criterion_logits = nn.LogSoftmax(dim=1).cuda(gpu)
    FL1=FocalLoss1()
    FL2=FocalLoss1()
    # Weighted_CE=weighted_ce()
    # writer = SummaryWriter(args.logs_dir)
    if process_rank == 0:
        save_checkpoint({
            'state_dict':net.module.state_dict(),
            'epoch': 0,}, False, 
            fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(0)))
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_lr(opt, epoch, args)

        # train(trainloader, net, criterion, opt, epoch, writer, args)
        # # init
        # train(trainloader, net, criterion, criterion_logits, opt, epoch, args)
        # log
        train(trainloader, net, criterion, criterion_nll, criterion_logits, FL1, FL2, opt, epoch, args)
        if process_rank == 0:
            save_checkpoint({
                'state_dict':net.module.state_dict(),
                'epoch': epoch+1,}, False,
                fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch+1)))
        

# def train(loader, net, crit, crit_log, opt, epoch, args):
def train(loader, net, crit, crit_nll, crit_log, FL1, FL2, opt, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    net.train()
    end = time.time()
    for i, ((feat, adj, cid, h1id), density, gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        feat, adj, cid, h1id, density, gtmat = map(lambda x: x.cuda(args.gpu, non_blocking=True),
                                (feat, adj, cid, h1id, density, gtmat))
        pred = net(feat, adj, h1id)
        labels = make_labels(gtmat).long()
        # m = nn.Softmax(dim=1)
        # loss = crit_nll(crit_log(pred), labels)
        if args.loss == 'normal':
            loss = normal(crit, pred, labels)
            loss = loss.mean()
        elif args.loss == 'ohem':
            loss = ohem(crit, pred, labels, args.ratio)
        elif args.loss == 'ohem3':
            if torch.sum(labels) == 0 or torch.sum(labels) == args.k_at_hop[0]:
                loss = normal(crit, pred, labels)
            else:
                loss = ohem3(crit, pred, labels)
        elif args.loss == 'class_balance_ce':
            loss = class_balance_ce(crit, pred, labels)
        elif args.loss == 'focal_loss':
            loss = focal_loss(crit, crit_nll, crit_log, pred, labels, args.gamma, args.alpha, normalize=False)
        elif args.loss == 'FL1':
            loss = FL1(pred, labels)
        elif args.loss == 'FL2':
            loss = FL2(pred, labels)
        elif args.loss == 'Weighted_CE':
            density = density.view(-1)
            # print(density)
            density[density < args.density_th] = 1
            # print(density)
            loss = weighted_ce(crit, pred, labels, density)
            # loss_ = normal(crit, pred, labels)
            loss = loss.mean()

        p, r, acc = accuracy(pred, labels)
        
        opt.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.1)
        opt.step()
        
        losses.update(loss.item(),feat.size(0))
        accs.update(acc.item(),feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r,feat.size(0))
    
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and dist.get_rank() == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        data_time=data_time, losses=losses, accs=accs, 
                        precisions=precisions, recalls=recalls))
            # if dist.get_rank() == 0:
            #     epoch_size = len(loader) // args.batch_size
            #     writer.add_scalar('train/loss', loss.item(), epoch * epoch_size + i)
            #     writer.add_scalar('train/acc', acc.item(), epoch * epoch_size + i)
            #     writer.add_scalar('train/precision', p, epoch * epoch_size + i)
            #     writer.add_scalar('train/recall', r, epoch * epoch_size + i)


def make_labels(gtmat):
    return gtmat.view(-1)


def adjust_lr(opt, epoch, args):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in args.lr_adjust_list:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale

    
def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 


if __name__ == '__main__':
    beg_time = time.time()
    main()
    print("==> train time use", time.time()-beg_time)
