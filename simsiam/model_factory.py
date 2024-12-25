from torch import nn
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class Linear(nn.Module):
    def __init__(self, nb_classes=10, feat=512):
        super(Linear, self).__init__()
        self.linear = nn.Linear(feat, nb_classes)

    def forward(self, x):
        return self.linear(x)


class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SimSiam(nn.Module):
    def __init__(self, emam, args):
        super(SimSiam, self).__init__()
        self.emam = emam  # 动量系数

        self.backbone = SimSiam.get_backbone(args.arch)  # 获取主干网络（哪个ResNet）
        self.backbone_k = SimSiam.get_backbone(args.arch)  # 获取动量主干网络
        dim_out, dim_in = self.backbone.fc.weight.shape  # 获取输出和输入维度
        dim_mlp = 2048  # MLP的维度
        self.backbone.fc = nn.Identity()  # 去掉全连接层
        self.backbone_k.fc = nn.Identity()  # 去掉动量全连接层

        print('dim in', dim_in)
        print('dim out', dim_out)
        self.projector = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))  # 投影头
        self.projector_k = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))  # 动量投影头

        # 预测头
        self.predictor = nn.Sequential(nn.Linear(dim_out, 512), BatchNorm1d(512), nn.ReLU(), nn.Linear(512, dim_out))

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )  # 编码器

        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )  # 动量编码器

        self.linear = Linear(nb_classes=args.nb_classes, feat=dim_in)  # 线性分类器
        self.probability = nn.Sequential(
            self.backbone,
            self.linear
        )  # 概率预测

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化动量编码器参数，使它和编码器参数完全一致
            param_k.requires_grad = False  # 动量编码器参数不更新

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152()}[backbone_name]  # 根据名称获取对应的ResNet模型

    def forward(self, im_aug1, im_aug2, img_weak):  # 两个强变换的图片，一个弱变换的图片
        output = self.probability(img_weak)  # 计算弱变换图片的概率

        z1 = self.encoder(im_aug1)  # 编码第一个强变换图片
        p1 = self.predictor(z1)  # 预测第一个强变换图片
        p1 = nn.functional.normalize(p1, dim=1)  # 每个像素点的通道归一化

        with torch.no_grad():  # no gradient to keys, 不污染
            m = self.emam  # 动量系数
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)  # 更新动量编码器参数

        z2 = self.encoder_k(im_aug2)  # 编码第二个强变换图片
        z2 = nn.functional.normalize(z2, dim=1)  # 每个像素点的通道归一化

        return p1, z2, output  # 返回预测结果和编码结果

    def forward_test(self, x):
        return self.probability(x)  # 测试时只返回概率





