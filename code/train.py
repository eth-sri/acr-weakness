# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import os

import argparse
import time
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, RandomSampler

import numpy as np
import random

from architectures import ARCHITECTURES
from datasets import get_dataset, DATASETS
from train_utils import AverageMeter, accuracy, log, test, requires_grad_, test_loss_log
from train_utils import prologue, seed_everything
from train_utils import p_correct

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_milestones',type=int, nargs='+', default=[50, 100],
                    help='milestones for MultiStepLR')

parser.add_argument('--lr_drop', type=int, default=1000,
                    help='When to drop the lr near the end of training')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')

parser.add_argument('--opt', type=str, choices=['sgd', 'adamw'], default='sgd')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--id', default=None, type=int,
                    help='experiment id, `randint(10000)` if None')
#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')

parser.add_argument('--num-noise-vec', default=4, type=int,
                    help="number of noise vectors. `m` in the paper.")

# one step discard
parser.add_argument("--ds_hard", action='store_true',
                    help='whether to ds heard samples at a point during training')
parser.add_argument("--ds_threshold", default=0.4, type=float,
                    help='discrad samples under this threshold')
parser.add_argument("--ds_epoch", default=100, type=int,
                    help='at which epoch to ds hard samples')
parser.add_argument("--p_corr_file", type=str, default=None,
                    help='path to p_correct file')

parser.add_argument("--dataset_weight", action='store_true',
                    help="use dataset weight")
parser.add_argument("--dataset_weight_epoch", default=10, type=int,
                    help='number of epochs for update dataset weight')
parser.add_argument("--num_noise_dataset_weight", default=16, type=int,
                    help='number of noise for dataset weight')

parser.add_argument("--adv", action='store_true',
                    help="apply adversarial attack")
parser.add_argument("--attacker", choices=['pgd_radius_l2', 'pgd_radius_linf'], default='pgd_radius_l2',
                    help='which attacker')
parser.add_argument("--attack_steps", default=1, type=int,
                    help='number of steps of attack')
parser.add_argument("--epsilon", default=256, type=float,
                    help="radius of PGD (Projected Gradient Descent) attack")

parser.add_argument("--checkpoint_freq", default=50, type=int)

args = parser.parse_args()

args.outdir = f"logs/{args.dataset}"
if args.adv:
    args.outdir = os.path.join(args.outdir, f"{args.epsilon}_{args.attack_steps}_{args.attacker}")

if args.ds_hard:
    args.outdir = os.path.join(args.outdir, f"ds_{args.ds_epoch}_{args.ds_threshold}")

args.outdir = os.path.join(args.outdir, f"num_{args.num_noise_vec}")

if args.dataset_weight:
    args.outdir = os.path.join(args.outdir, f"dataset_weight_{args.dataset_weight_epoch}")

args.outdir = os.path.join(args.outdir, f"noise_{args.noise_sd}")


def main():
    seed = args.id
    seed_everything(seed)

    train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)

    if args.ds_hard:
        ds_epochs = [args.ds_epoch]
    else:
        ds_epochs = [0]

    if args.dataset_weight:
        update_weight_epochs = [i for i in range(ds_epochs[-1], args.epochs, args.dataset_weight_epoch)]
    else:
        update_weight_epochs = [args.epochs + 1]

    train_dataset = get_dataset(args.dataset, 'train')
    args.dataset_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                num_workers=args.workers)

    if args.adv:
        if args.attacker == 'pgd_radius_l2':
            attacker = radius_PGD(steps=args.attack_steps, step_size=args.epsilon, type='l2', device=device)
        elif args.attacker == 'pgd_radius_linf':
            attacker = radius_PGD(steps=args.attack_steps, step_size=args.epsilon, type='linf', device=device)
        else:
            raise NotImplementedError
    else:
        attacker = None

    args.adv_active = False

    easy_indices = np.arange(args.dataset_num)

    for epoch in range(starting_epoch, args.epochs):        
        if args.adv and epoch >= ds_epochs[-1]:
            args.adv_active = True
        
        outdir_p_corr = args.outdir.replace("logs", "p_correct/train")
        outdir_p_corr = os.path.join(outdir_p_corr, f"epoch")

        if args.ds_hard and epoch in ds_epochs:
            threshold = args.ds_threshold
            if not args.p_corr_file:
                N0 = 100
                batch_size = 64
                args.p_corr_file = os.path.join(outdir_p_corr, f'{epoch - 1}_{N0}.npy')

                p_correct(model, epoch - 1, args.noise_sd, outdir_p_corr, N0, args.dataset, 
                            "train", batch_size=batch_size)

            p_corr = np.load(args.p_corr_file)
            easy_indices = np.where(p_corr >= threshold)[0]

            train_dataset = Subset(train_dataset, easy_indices)

        print(f"number of remained data: {len(easy_indices)}")
        

        if args.dataset_weight:
            if epoch in update_weight_epochs:
                args.p_corr_file_weight = os.path.join(outdir_p_corr, f'{epoch - 1}_{args.num_noise_dataset_weight}.npy')
                batch_constant = 8192
                p_correct(model, epoch - 1, args.noise_sd, outdir_p_corr, args.num_noise_dataset_weight, args.dataset, 
                            "train", batch_size=int(batch_constant / args.num_noise_dataset_weight), valid_mask=easy_indices)

                p_corr_light = np.load(args.p_corr_file_weight)

                corr_num = (np.clip(p_corr_light, a_min=0.75, a_max=1.0) * args.num_noise_dataset_weight).astype(int)

                weight = norm.ppf(proportion_confint(corr_num, args.num_noise_dataset_weight, alpha=2 * 0.1, method="beta")[0])

            if epoch >= update_weight_epochs[0]:
                sampler = WeightedRandomSampler(weight, args.dataset_num, replacement=True)

        else:
            if epoch >= ds_epochs[-1]:
                sampler = RandomSampler(train_dataset, replacement=True, num_samples=args.dataset_num)
            else:
                sampler = None
        
        if epoch >= ds_epochs[-1]:
            shuffle = True if sampler is None else None
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=shuffle,
                                        num_workers=args.workers, sampler=sampler)

        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd, attacker, device, writer)
        scheduler.step()
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args.noise_sd, device, writer, args.print_freq)
        after = time.time()

        # train_mean_loss_str = ",".join("{:.3f}".format(x) for x in train_mean_loss)
    
        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
            epoch, after - before,
            scheduler.get_last_lr()[0], train_loss, train_acc, test_loss, test_acc, len(train_dataset)))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        torch.save({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoints_path = model_path.replace('.pth.tar', f's/checkpoint{epoch}.pth.tar')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoints_path)


def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]



def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer,
          epoch: int, noise_sd: float, attacker, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)

        for inputs, targets in mini_batches:

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            model.eval()

            noises = [torch.randn_like(inputs, device=device) * noise_sd # * noise_scale
                for _ in range(args.num_noise_vec)]

            if args.adv_active:
                inputs_attack = inputs
                targets_attack = targets
                noises_attack = noises

                requires_grad_(model, False)
                noises_attacked = attacker.attack(model, inputs_attack, targets_attack, noises=noises_attack)
                requires_grad_(model, True)

                noises = noises_attacked

            model.train()

            inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
            targets_r = targets.repeat(args.num_noise_vec)

            # compute output
            outputs = model(inputs_c)

            criterion_nr = CrossEntropyLoss(reduction='none').to(device)
            loss_ce = criterion_nr(outputs, targets_r)
            
            loss = loss_ce.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets_r, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # time.sleep(5)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    if writer:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


class radius_PGD(object):
    """
    PGD attack, every step project the noise on the 

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(self,
                 steps: int,
                 step_size: float,
                 type: str = 'l2',
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(radius_PGD, self).__init__()
        self.steps = steps
        self.step_size = step_size
        self.random_start = random_start
        self.type = type
        self.max_norm = max_norm
        self.device = device

    def attack(self, model, inputs, labels, noises=None):
        """
        Performs PGD attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Smoothed predictions of the samples to attack.
        noises : List[torch.Tensor]
            Lists of noise samples to attack.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        m = len(noises)
        inputs_r = inputs.repeat(m, 1, 1, 1)
        labels_r = labels.repeat(m)
        noise0 = torch.cat(noises, dim=0)
        noise = noise0.detach()
        batch_size = inputs_r.size(0)

        noise0_norm = _batch_l2norm(noise0).view(-1, 1, 1, 1)

        noise_best = torch.zeros_like(noise)
        best_indices = torch.ones(batch_size, dtype=torch.bool, device=device) * self.steps
        is_adv = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(self.steps):
            noise.requires_grad_()
            logits_r = model(inputs_r + noise)

            pred_labels = torch.argmax(logits_r, dim=1)

            new_best = (pred_labels != labels_r) & (best_indices == self.steps)
            best_indices[new_best] = i

            noise_best[new_best] = noise[new_best]

            criterion = CrossEntropyLoss(reduction='none').to(device)

            loss_ce = criterion(logits_r, labels_r)


            loss = loss_ce.sum() / m

            grad = torch.autograd.grad(loss, [noise])[0]

            if self.type == 'l2':
                grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
                grad = grad / (grad_norm + 1e-8)
                noise = noise + self.step_size * grad

            elif self.type == 'linf':
                grad = grad.sign()
                noise = noise + self.step_size * grad / len(inputs[0].view(-1)) ** 0.5
        
            noise_norm = _batch_l2norm(noise).view(-1, 1, 1, 1)
            noise = noise / (noise_norm + 1e-8) * noise0_norm


        new_best = best_indices == self.steps
        noise_best[new_best] = noise[new_best]

        return torch.chunk(noise_best, m, dim=0)


if __name__ == "__main__":
    main()