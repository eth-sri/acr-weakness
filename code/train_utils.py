import os
import shutil
import time
import datetime

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from third_party.core import Smooth
from tqdm import tqdm

import copy

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from architectures import get_architecture
from datasets import get_dataset, get_num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed, strict=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ppf_weight(prob):
    device = prob.device
    prob_clamped = torch.clamp(prob, 0.6, 0.9999).cpu().numpy()
    return torch.tensor(norm.ppf(prob_clamped)).to(device)


def dataset_weight(NA, N):
    device = prob.device
    prob_clamped = torch.clamp(prob, 0.75, 1.0).cpu().numpy()
    return torch.tensor(1 / (prob_clamped * 10)).to(device)

def set_bn_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    return model


def p_correct(model, epoch:int, sigma:float, outdir:str, N0:int, dataset:str, split:str, batch_size:int=1000, 
                valid_mask=None, train:bool=False, num_workers:int=4):
    print(f"p_correct working on epoch {epoch}")

    n_classes = get_num_classes(dataset)
    dataset = get_dataset(dataset, split)
    if valid_mask is not None:
        dataset = Subset(dataset, valid_mask)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    outfile = os.path.join(outdir, f'{epoch}_{N0}.csv')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(outfile):
        raise 'file already exists'
    f = open(outfile, 'w')
    print("idx\tlabel\tconfidence", file=f, flush=True)
    

    confidences = np.ones((len(dataset),)) / n_classes

    model.eval()

    confidences = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            x, label = batch
            x = x.to(device)
            target = label.to(device)

            inputs = x.repeat(N0, 1, 1, 1)
            target = target.repeat(N0)
            noise = torch.randn_like(inputs, device=device) * sigma

            output = model(inputs + noise)
            prediction = output.argmax(dim=1).view((N0, -1))

            confidence = (prediction == target.view((N0, -1))).float().mean(dim=0)

            confidences.append(confidence.cpu().numpy())

            for j in range(len(confidence)):
                print("{}\t{}\t{:.2}".format(i * batch_size + j, label[j], confidence[j].cpu().item()), file=f, flush=True)

    confidences = np.concatenate(confidences, axis=0)

    f.close()

    np.save(os.path.join(outdir, f'{epoch}_{N0}.npy'), confidences)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))


def prologue(args, use_arch2=False, use_arch3=False):
    if not hasattr(args, 'id') or args.id is None:
        args.id = np.random.randint(10000)

    args.outdir = args.outdir + f"/{args.arch}/{args.id}/"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        os.makedirs(args.outdir + "/checkpoints")

    # Copies files to the outdir to store complete script with each experiment
    copy_code(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    print("dataset get finish")
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    print("dataset loader finish")

    

    print("create the model")
    model = get_architecture(args.arch, args.dataset)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc\tremain data")
    writer = SummaryWriter(args.outdir)

    criterion = CrossEntropyLoss().to(device)
    params = model.parameters()
    print(args.lr)
    if args.opt == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    milestones = args.lr_milestones
    milestones.append(args.lr_drop)

    starting_epoch = 0

    if args.pretrained_model != '':
        if os.path.isfile(args.pretrained_model):
            print("=> loading pretrained model '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded pretrained model '{}'".format(args.pretrained_model))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained_model))

    # Load latest checkpoint if exists (to handle philly failures)
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            starting_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma, last_epoch=starting_epoch - 1)

    return train_loader, test_loader, criterion, model, optimizer, scheduler, \
           starting_epoch, logfilename, model_path, device, writer

def test_loss_log(loader, model, criterion, epoch, noise_sd, device, writer=None, print_freq=10, num_log_loss_interval=5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    loss_confidence_dict = {new_list: [] for new_list in range(num_log_loss_interval)}

    confidence_bins = np.arange(0, 1+1e-3, 1. / num_log_loss_interval)
    confidence_bins[-1] = 1+1e-3

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets, pb = batch
            inputs, targets, pb = inputs.to(device), targets.to(device), pb.to(device)
            
            pb_confidence = pb[torch.arange(0, pb.size(0)), targets]
            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd

            # compute output
            outputs = model(inputs)
            criterion_nr = CrossEntropyLoss(reduction='none').to(device)
            loss = criterion_nr(outputs, targets)

            bins_edges = np.searchsorted(np.sort(pb_confidence.cpu().detach().numpy()), confidence_bins)


            pb_loss = np.stack((pb_confidence.cpu().detach().numpy(), loss.cpu().detach().numpy()), axis=0)
            sorted_idx = np.argsort(pb_confidence.cpu().detach().numpy())
            pb_loss = pb_loss[:, sorted_idx]

            for interval_i in range(num_log_loss_interval):
                if loss_confidence_dict[interval_i] is None:
                    loss_confidence_dict[interval_i] = pb_loss[1, bins_edges[interval_i]:bins_edges[interval_i + 1]]
                else:
                    loss_confidence_dict[interval_i].extend(pb_loss[1, bins_edges[interval_i]:bins_edges[interval_i + 1]])

            loss = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        mean_loss = []
        for interval_i in range(num_log_loss_interval):
            loss_i = loss_confidence_dict[interval_i]
            if len(loss_i) != 0:
                mean_loss.append(sum(loss_i) / len(loss_i))
            else:
                mean_loss.append(0)

        return (losses.avg, top1.avg, mean_loss)


def test(loader, model, criterion, epoch, noise_sd, device, writer=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)
    

def test_base(loader, model, criterion, epoch, device, writer=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)


def normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def check_spectral_norm(m, name='weight'):
    from torch.nn.utils.spectral_norm import SpectralNorm
    for k, hook in m._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            return True
    return False


def apply_spectral_norm(m):
    from torch.nn.utils import spectral_norm
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            spectral_norm(layer)
        elif isinstance(layer, nn.Linear):
            spectral_norm(layer)
        elif isinstance(layer, nn.Embedding):
            spectral_norm(layer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

