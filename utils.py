import math
import torch
from PIL import ImageFilter
import random

import torch


class LatentThreeCropsTransformSame:
    """在隐空间中生成三个变换后的版本"""
    def __init__(self, noise_std=0.1, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, mean, std):
        # 原始版本
        crop1 = mean.clone()
        crop2 = mean.clone()
        crop3 = mean.clone()

        return crop2, crop3, crop1
    
class LatentThreeCropsTransform:
    """在隐空间中生成三个变换后的版本"""
    def __init__(self, noise_std=0.1, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, mean, std):
        # 原始版本
        crop1 = mean.clone()

        # 增广版本1：添加高斯噪声并且缩放
        noise = torch.randn_like(mean) * self.noise_std
        scale = torch.empty_like(mean).uniform_(*self.scale_range)
        crop2 = (mean + noise)*scale
        
        # 增广版本1：相同进行增广
        noise = torch.randn_like(mean) * self.noise_std
        scale = torch.empty_like(mean).uniform_(*self.scale_range)
        crop3 = (mean + noise)*scale

        return crop2, crop3, crop1

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class ThreeCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, base_transform2):
        self.base_transform = base_transform
        self.base_transform2 = base_transform2

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        p = self.base_transform2(x)
        return [q, k, p]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def save_checkpoint(epoch, model, model_ema, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'ema_state_dict': model_ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_checkpoint(model, model_ema, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model_ema.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, model_ema, optimizer