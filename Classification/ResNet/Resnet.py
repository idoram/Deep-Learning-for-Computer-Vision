import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ds = downsample

    def forward(self, x):
        identity = x
        if self.ds:
            identity = self.ds(x)

        out = self.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param out_channels: 输出的通道数，这里是与输入一致的，指的是第二层的卷积核个数（resnet的 50，101，152层的结构）
        """
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.ds = downsample

    def forward(self, x):
        identity = x
        if self.ds:
            identity = self.ds(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn1(self.conv2(out)))
        out = self.bn2(self.conv3(out))

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        """
        :param block: 使用的是 BasicBlock还是 BottleNeck
        :param blocks_num: 列表结构，传入残差结构的数目
        :param num_classes: 训练集的分类个数
        :param include_top: 方便在 ResNet基础上搭建更复杂的网络
        :param groups: 卷积操作中的分组数
        :param width_per_group: 卷积操作中每个分组的通道数
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64  # 第一次MaxPooling后的通道数

        self.group = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layers(block, 64, blocks_num[0])
        self.layer2 = self.make_layers(block, 128, blocks_num[1], stride=2)
        self.layer3 = self.make_layers(block, 256, blocks_num[2], stride=2)
        self.layer4 = self.make_layers(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgp = nn.AdaptiveAvgPool2d((1, 1))
            self.dense = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, block, channels, block_num, stride=1):
        ds = None
        if stride != 1 or self.in_channels != block.expansion * channels:
            ds = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(chennels * block.expansion)
            )

        layers = [block(self.in_channels, channels, downsample=ds, stride=stride, group=self.group,
                        width_per_group=self.width_per_group)]
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels, groups=self.group, width_per_group=self.width_per_group))

        # * 被用作解包操作符，将列表 layers 拆解为多个独立的参数，相当于将列表中的每个元素当作一个独立的参数传递给函数。
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxp(self.relu(self.bn(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.include_top:
            out = self.avgp(out)
            out = torch.flatten(out, 1)
            out = self.dense(out)

        return out


# 如要使用迁移学习，记得使用相同的数据预处理方法
def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
