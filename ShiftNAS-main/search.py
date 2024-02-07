import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
from radam import RAdam
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search import Network
from architect import Architect
import random

parser = argparse.ArgumentParser("cifar")
# shift setting
parser.add_argument('--shift_type', type=str, default='PS', choices=['Q', 'PS'], help='type of shift method')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'],
                    help='type of rounding (default: deterministic)')
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--activation_bits', nargs='+', default=[16,16],
                    help='number of integer and fraction bits to represent activation (fixed point format)')
# model setting
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--op_search_layers', type=int, default=8, help='total number of layers')
parser.add_argument('--tp_search_layers', type=int, default=20, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# training setting
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar100', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=70, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--lr_sign', type=float, default=None, help='init learning rate for sign')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_lr', type=float, default=1e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# save setting
parser.add_argument('--save', type=str, default='C100', help='experiment name')
# parser.add_argument('--seed', type=int, default=random.randint(1,10000), help='random seed')
parser.add_argument('--seed', type=int, default=8029, help='random seed')
parser.add_argument('--note', type=str, default='radam', help='note of running')
#anneal weight
parser.add_argument('--Tmax', type=float, default=10, help='learning rate for arch encoding')
parser.add_argument('--Ttpmin', type=float, default=0.02, help='weight decay for anneal topology')
parser.add_argument('--Op_Pretrain_Start', type=int, default=0, help='learning rate for arch encoding')
parser.add_argument('--Op_Search_Start', type=int, default=15, help='weight decay for arch encoding')
parser.add_argument('--Tp_Pretrain_Start', type=int, default=30, help='learning rate for arch encoding')
parser.add_argument('--Tp_Search_Start', type=int, default=50, help='weight decay for arch encoding')
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
# print(device)

args.save = 'logs/search/search-{}-{}-{}'.format(args.save, args.seed, args.note)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100

#define four stages
Op_Pretrain_Start=args.Op_Pretrain_Start
Op_Search_Start=args.Op_Search_Start
Tp_Pretrain_Start=args.Tp_Pretrain_Start
Tp_Search_Start=args.Tp_Search_Start

Tp_Anneal_Rate=pow(args.Ttpmin/args.Tmax,1.0/(args.epochs-Tp_Search_Start-1)) ## The value of \theta


def print_genotype(model):
    genotype, weight_normal, weight_reduce = model.genotype()

    logging.info('genotype = %s', genotype)
    logging.info("normal:")
    logging.info(weight_normal)
    logging.info("reduce:")
    logging.info(weight_reduce)
    if 'tp' in model.phase:
        normal_edge_dict, reduce_edge_dict = model.parse_edge()
        logging.info("normal edges:")
        logging.info(normal_edge_dict)
        logging.info("reduce edges:")
        logging.info(reduce_edge_dict)

def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    model = Network(args.init_channels, CIFAR_CLASSES, args.op_search_layers, criterion)
    start_epoch=0
    model = model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model_other_params = []
    model_sign_params = []
    model_shift_params = []
    for name, param in model.named_parameters():
        if(name.endswith(".sign")):
            model_sign_params.append(param)
        elif(name.endswith(".shift")):
            model_shift_params.append(param)
        else:
            model_other_params.append(param)
    params_dict = [
        {"params": model_other_params},
        {"params": model_sign_params, 'lr': args.lr_sign if args.lr_sign is not None else args.lr, 'weight_decay': 0},
        {"params": model_shift_params, 'lr': args.lr, 'weight_decay': 0}
        ]

    optimizer = None
    if(args.shift_type == 'PS'):
        optimizer = RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.shift_type == 'Q'):
        optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.lr_min)
     
    arch_optimizer = torch.optim.Adam(model.arch_parameters(),
                                        lr=args.arch_lr, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

    architect = Architect(model, args)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    if args.set=='cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        val_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        val_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue_A = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    train_queue_B = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    train_queue_Full = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.workers)

    for epoch in range(start_epoch, args.epochs):
        if epoch == Op_Pretrain_Start:
            model.phase = 'op_pretrain'
            logging.info("Begin operation pretrain!")
        elif epoch == Op_Search_Start:
            model.phase = 'op_search'
            logging.info("Begin operation search!")
        elif epoch == Tp_Pretrain_Start:
            model.__init__(args.init_channels,
                            CIFAR_CLASSES, args.op_search_layers, criterion, init_arch=False)
            model.phase = 'tp_pretrain'

            model_other_params = []
            model_sign_params = []
            model_shift_params = []
            for name, param in model.named_parameters():
                if(name.endswith(".sign")):
                    model_sign_params.append(param)
                elif(name.endswith(".shift")):
                    model_shift_params.append(param)
                else:
                    model_other_params.append(param)
            params_dict = [
                {"params": model_other_params},
                {"params": model_sign_params, 'lr': args.lr_sign if args.lr_sign is not None else args.lr, 'weight_decay': 0},
                {"params": model_shift_params, 'lr': args.lr, 'weight_decay': 0}
                ]
            if(args.shift_type == 'PS'):
                optimizer = RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
            elif(args.shift_type == 'Q'):
                optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.lr_min)
            
            model.prune_model()
            arch_optimizer = torch.optim.Adam(model.arch_parameters(),
                                                lr=args.arch_lr, betas=(0.9, 0.999),
                                                weight_decay=args.arch_weight_decay)
            model = model.to(device)
            architect = None  # use one-step to optimize topology
            logging.info("Prune model finish!")
            logging.info("Load Prune Architecture finish!")
            logging.info("Begin topology pretrain!")
        elif epoch == Tp_Search_Start:
            model.phase = 'tp_search'
            logging.info("Begin topology search!")
        else:
            pass

        if 'pretrain' in model.phase:
            model.T = 1.0
        else:
            if 'op' in model.phase:
                model.T = 1.0
            else:
                model.T = 10 * pow(Tp_Anneal_Rate, epoch - Tp_Search_Start)

        scheduler.step(epoch)
        # scheduler.step()

        lr = scheduler.get_last_lr()[0]
        logging.info('epoch:%d phase:%s lr:%e', epoch, model.phase, lr)

        print_genotype(model)

        # training
        if 'op' in model.phase:
            train_acc, train_obj = train_op(train_queue_A, train_queue_B, model, architect, criterion, optimizer, lr)
        else:
            train_acc, train_obj = train_tp(train_queue_A, train_queue_Full, model, criterion, optimizer,arch_optimizer)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # utils.save(model, os.path.join(args.save, 'weights_%s.pth'%model.phase))
        # model.save_arch(os.path.join(args.save, 'arch_%s.pth'%model.phase))

    print_genotype(model)

def shift_l2_norm(opt, weight_decay):
    shift_params = opt.param_groups[2]['params']
    l2_norm = 0
    for shift in shift_params:
        l2_norm += torch.sum((2**shift)**2)
    return weight_decay * 0.5 * l2_norm

def train_op(train_queue_A, train_queue_B, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue_A):
        model.train()
        n = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # get a random minibatch from the search queue with replacement
        #input_search, target_search = next(iter(valid_queue))
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(train_queue_B)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.to(device, non_blocking=True)
        target_search = target_search.to(device, non_blocking=True)

        if 'pretrain' not in model.phase:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)
        # if(args.weight_decay > 0):
        #    loss += shift_l2_norm(optimizer, args.weight_decay)

        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train iter:%03d T:%4f loss:%e top1:%f top5:%f', step,model.T, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def train_tp(train_queue_A, train_queue_Full, model, criterion, optimizer, arch_optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_queue=train_queue_A if 'pretrain' in model.phase else train_queue_Full

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if 'pretrain' not in model.phase:
            arch_optimizer.zero_grad()

        optimizer.zero_grad()

        logits = model.forward_tp(input)
        loss = criterion(logits, target)
        # if(args.weight_decay > 0):
        #    loss += shift_l2_norm(optimizer, args.weight_decay)

        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        if 'pretrain' not in model.phase:
            arch_optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train iter:%03d T:%4f loss:%e top1:%f top5:%f', step, model.T, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if 'tp' in model.phase:
                logits = model.forward_tp(input)
            else:
                logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
  main() 



