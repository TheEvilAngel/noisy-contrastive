import argparse
import time
import math
from os import path, makedirs
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import copy
import torch.nn as nn
from simsiam.model_factory import SimSiam
import pandas as pd


from loader import CIFAR10N, CIFAR100N, VAEAugmentedDataset
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint, accuracy, load_checkpoint, ThreeCropsTransform, LatentThreeCropsTransform


parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default='./data', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default='./save', type=str, help='path to experiment directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='path to dataset', choices=["cifar10", "cifar100", "imagenet10", "imagenet100"])
parser.add_argument('--noise_type', default='sym', type=str, help='noise type: sym or asym', choices=["sym", "asym"])
parser.add_argument('--r', type=float, default=0.8, help='noise level')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=28, type=int)

parser.add_argument('--arch', default='resnet18', help='model name is used for training')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=550, help='number of training epochs')

parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--m', type=float, default=0.99, help='moving average of probbility outputs')
parser.add_argument('--tau', type=float, default=0.8, help='contrastive threshold (tau)')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--lamb', default=50.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--type', default='ce', type=str, help='ce or gce loss', choices=["ce", "gce"])
parser.add_argument('--beta', default=0.6, type=float, help='gce parameter')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpu', default=5, type=int, help='GPU id to use.')

args = parser.parse_args()
import random
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.dataset == 'cifar10':
    args.nb_classes = 10
elif args.dataset == 'cifar100':
    args.nb_classes = 100
elif args.dataset == 'imagenet100':
    args.nb_classes = 50
elif args.dataset == 'imagenet10':
    args.nb_classes = 50


class GCE_loss(nn.Module):
    # 定义GCE_loss类，继承自nn.Module
    def __init__(self, q=0.8):
        # 初始化函数，设置q参数
        super(GCE_loss, self).__init__()
        self.q = q

    def forward(self, outputs, targets):
        # 前向传播函数
        targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
        # 将targets转换为one-hot编码
        pred = F.softmax(outputs, dim=1)
        # 对outputs进行softmax操作
        pred_y = torch.sum(targets * pred, dim=1)
        # 计算预测值与真实值的乘积
        pred_y = torch.clamp(pred_y, 1e-4)
        # 对pred_y进行clamp操作，防止出现0值，最小值为1e-4
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        # 计算最终的损失 Σ (1 - p(y|x)^q) / q
        # q=1, MAE loss; q=0, CE loss，Beta表示
        return final_loss
        # 返回最终的损失



if args.type == 'ce':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = GCE_loss(args.beta)


def set_model(args):
    model = SimSiam(args.m, args)
    model.cuda(args.gpu) # 也要是对应的gpu
    return model

# 2. 加载数据
def set_loader(args):
    # 加载VAE生成的数据
    train_data = torch.load(path.join(args.data_root, 'train.pt'))
    test_data = torch.load(path.join(args.data_root, 'test.pt'))

    # 提取字段
    train_mean, train_std, train_labels = train_data['mean'], train_data['std'], train_data['labels']
    test_mean, test_std = test_data['mean'], test_data['std']

    # 更改精度
    train_mean, train_std = train_mean.to(torch.float32), train_std.to(torch.float32)
    test_mean, test_std = test_mean.to(torch.float32), test_std.to(torch.float32)
    
    # 构造隐空间增广
    latent_transform = LatentThreeCropsTransform(noise_std=0.05, scale_range=(0.95, 1.05))

    # 构造训练和测试集
    train_set = VAEAugmentedDataset(train_mean, train_std, train_labels, transform=latent_transform)
    test_set = VAEAugmentedDataset(test_mean, test_std)

    # DataLoader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

def train(train_loader, model, criterion, optimizer, epoch,  args):
    batch_time = AverageMeter('Time', ':6.3f') # 定义输出格式
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train() # 变成训练模式

    end = time.time()
    for i, (images, targets, index) in enumerate(train_loader):
        bsz = targets.size(0) # batch size
        
        # images为ThreeCropsTransform中生成的三个图片
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        # p1=encoder+predict(images[0])；z2=encoder_k(images[1])；outputs=probability(images[2])
        p1, z2, outputs = model(images[0], images[1], images[2])


        # avoid collapsing and gradient explosion
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)

        contrast_1 = torch.matmul(p1, z2.t())  # B X B 假设出来就是B*B的矩阵，计算p1和z2的相似度

        '''论文核心'''
        # <q,z> + log(1-<q,z>) 
        # 对角线全是-contrast_1 + 除了对角线全是(1-contrast_1).log()
        # contrast_1 = -contrast_1*torch.zeros(bsz, bsz).fill_diagonal_(1).cuda() + ((1-contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda()
        contrast_1 = (-2+1e-4)*torch.zeros(bsz, bsz).fill_diagonal_(1).cuda() + ((1-contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda()
        contrast_logits = 2 + contrast_1


        soft_targets = torch.softmax(outputs, dim=1) # 最终probability输出的概率分布
        contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach() # 创建一个新的张量，并断开其梯度计算
        contrast_mask.fill_diagonal_(1) # 样本和样本自己的相似度是1
        pos_mask = (contrast_mask >= args.tau).float() # 高于tau的相似度保留，低于tau的相似度置为0
        contrast_mask = contrast_mask * pos_mask # 保留高相似度的样本
        contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True) # 对每行元素归一化（把每一列看成一个整体）
        loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)

        loss_ce = criterion(outputs, targets) # CE/GCE loss


        loss = args.lamb*loss_ctr + loss_ce

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        # losses = AverageMeter('Loss', ':.4e')
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    return losses.avg


def predict_test_set(test_loader, model):
    model.eval()
    results = []
    
    with torch.no_grad():
        for features, index in test_loader:
            if args.gpu is not None:
                features = features.cuda(args.gpu, non_blocking=True)
            outputs = model.forward_test(features)  # 前向推理
            predicted_labels = outputs.argmax(dim=1).cpu().numpy()  # 获取预测标签
            index = index.numpy()  # 转换索引为numpy
            for idx, label in zip(index, predicted_labels):
                results.append({"index": idx, "labels": label})
    
    return results


def save_predictions_to_csv(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def load_model(model_path, args):
    """加载保存的模型"""
    model = set_model(args)  # 初始化模型
    checkpoint = torch.load(model_path, map_location=f"cuda:{args.gpu}")
    model.load_state_dict(checkpoint['state_dict'])
    # model = model.to(torch.bfloat16)
    model.cuda(args.gpu)
    model.eval()  # 设置为评估模式
    print(f"Loaded model from {model_path}")
    return model


def save_model(epoch, model, optimizer, best_loss, filename, msg):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(state, filename)
    print(msg)
    
def create_exp_dir(base_dir="./save"):
    """创建基于时间戳的实验文件夹"""
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + 'third-'+ args.dataset + '-' + args.type + '-' + args.noise_type
    exp_dir = path.join(base_dir, timestamp)
    makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir, timestamp

def main():
    print(vars(args))

    exp_dir, timestamp = create_exp_dir(args.exp_dir)
    
    train_loader, test_loader = set_loader(args)

    model = set_model(args)
    
    '''模型转换为bfloat16'''
    # model = model.to(torch.bfloat16)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

    start_epoch = 0

    best_loss = float('inf')  # 初始化为正无穷
    last_model_path = path.join(exp_dir, "last_model.pth")

    for epoch in range(start_epoch, args.epochs):
        epoch_optim = epoch

        adjust_learning_rate(optimizer, epoch_optim, args)
        print("Training...")

        # train for one epoch
        time0 = time.time()
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        print("Train \tEpoch:{}/{}\ttime: {}\tLoss: {}".format(epoch, args.epochs, time.time()-time0, train_loss))
        
        if epoch > 100 and epoch % 10 == 0:
            model_path = path.join(exp_dir, f"model_{epoch}.pth")
            save_model(epoch, model, optimizer, train_loss, model_path, msg="Model saved in {} epoch".format(epoch))
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
    
    save_model(epoch, model, optimizer, train_loss, last_model_path, msg="Last model saved in {} epoch".format(epoch))
    
    # 对每个模型进行预测
    for epoch in range(0, args.epochs, 1):
        print("Predicting using the {} model.", format(epoch))
        model_path = path.join(exp_dir, f"model_{epoch}.pth")
        model = load_model(model_path, args)
        results = predict_test_set(test_loader, model)
        save_predictions_to_csv(results, path.join(exp_dir, f"model_{epoch}_predictions.csv"))
    
    # 使用最佳模型进行预测
    print("Predicting using the best model...")
    best_model_path = path.join(exp_dir, f"model_{best_epoch}.pth")
    best_model = load_model(best_model_path, args)
    best_results = predict_test_set(test_loader, best_model)
    save_predictions_to_csv(best_results, path.join(exp_dir, timestamp+f"model_{epoch}"+"-best_model_predictions.csv"))
    print("Best epoch is {}.", format(best_epoch))

    # 使用最后的模型进行预测
    print("Predicting using the last model...")
    last_model = load_model(last_model_path, args)
    last_results = predict_test_set(test_loader, last_model)
    save_predictions_to_csv(last_results, path.join(exp_dir, timestamp+"-last_model_predictions.csv"))

    with open('log.txt', 'a') as f:
        if args.type == 'ce':
            f.write('dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: ce \t seed: {}\n'.format(args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.seed))
        elif args.type == 'gce':
            f.write('dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {}\n'.format(args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed))


if __name__ == '__main__':
    main()



