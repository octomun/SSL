import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from PIL import ImageFilter
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/MONG/benchmark/ImageNet/ILSVRC/Data/ImageNet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float, metavar='LR',
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=500, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=True, type=bool)
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPU id to use.')

parser.add_argument('--projector', default='8192-8192-8192', type=str, metavar='MLP', help='projector MLP')
parser.add_argument('--tau', default=0.1, type=float, help='temperature (default: 0.1)')
parser.add_argument('--dim', default=2048, type=int, help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int, help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true', help='Fix learning rate for the predictor')
parser.add_argument('--checkpoint', default='./checkpoint', type=Path, metavar='DIR',
                    help='path to checkpoint directory')


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.ngpus_per_node = len(args.gpu.split(','))
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    args.distributed = True

    if args.rank == 0:
        args.checkpoint.mkdir(parents=True, exist_ok=True)
    print('start')
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = SimSiam(args)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    model.cuda(args.gpu)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    per_batch_size = int(args.batch_size / args.ngpus_per_node)
    args.workers = args.ngpus_per_node
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if (args.checkpoint / 'checkpoint.pth').is_file():
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint / 'checkpoint.pth', map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, '{}/checkpoint.pth'.format(checkpoint['epoch'])))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        Solarization(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_dataset = datasets.ImageFolder(traindir, TwoCropsTransform(transforms.Compose(augmentation)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_batch_size, num_workers=args.workers, drop_last=True,
                                               pin_memory=True, sampler=train_sampler)


    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)
        if args.rank == 0:
            save_model(args, args.checkpoint, epoch, model.module)

    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(), args.checkpoint / 'resnet50.pth')

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    print('=>  train start')
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output and loss
        loss = model(images[0], images[1])

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0 and args.gpu == 0:
            progress.display(i)



def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.encoder = torchvision.models.resnet50(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        loss = self.sim(z1.detach(), p2) / 2 + self.sim(z2.detach(), p1) / 2
        return -loss.mean()

    def sim(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.sum(x * y, dim=1, keepdim=True)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k

def save_model(args, checkpoint, epoch, model):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict()
        },
        f=checkpoint+'/checkpoint.pth'
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()