import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import genotypes
from radam import RAdam
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
from model import NetworkImageNet as Network
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("cifar")
# shift setting
parser.add_argument('--shift_type', type=str, default='PS', choices=['Q', 'PS'], help='type of shift method')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'],
                    help='type of rounding (default: deterministic)')
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--activation_bits', nargs='+', default=[16,16],
                    help='number of integer and fraction bits to represent activation (fixed point format)')
# model setting
parser.add_argument('--init_channels', type=int, default=46, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
# training setting
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar100', help='location of the data corpus')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate') # 0.1 or 0.01
parser.add_argument('--lr_sign', type=float, default=None, help='init learning rate for sign')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') # 1e-4 or 3e-4
parser.add_argument('--optimizer', type=str, default='radam', choices=['sgd', 'radam'], help='optimizer algorithm')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# save setting
parser.add_argument('--checkpoint', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='shiftNAS', help='experiment name')
parser.add_argument('--seed', type=int, default=random.randint(1,1000), help='random seed')
parser.add_argument('--note', type=str, default='radam', help='note of running')
parser.add_argument('--arch', type=str, default='shiftNAS_final_C100', help='which architecture to use')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
# print(device)

args.save = 'logs/eval/eval-{}-{}-{}-{}'.format(args.save,args.set,args.seed,args.note)
utils.create_exp_dir(args.save, scripts_to_save=None)
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info(genotype)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
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
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.to(device)

    optimizer = None
    if(args.shift_type == 'PS'):
        optimizer = RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.shift_type == 'Q'):
        optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    

    traindir = os.path.join(args.data_dir, 'train')
    validdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # if args.set=='cifar100':
    #     train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    #     valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    # else:
    #     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    #     valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 180])
    lr = args.lr
    for epoch in range(args.epochs):       
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        scheduler.step()

        utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

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
            logits,_ = model(input)
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







