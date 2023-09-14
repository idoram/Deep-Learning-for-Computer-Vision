import torch
import torch.nn as nn


# 输入图像 -> 卷积层(3x3卷积核, 填充1, 步长1) -> ReLU激活函数 -> 池化层(2x2池化核, 步长2) -> 卷积层(3x3卷积核, 填充1, 步长1) ->
# ReLU激活函数 -> 池化层(2x2池化核, 步长2) -> 卷积层(3x3卷积核, 填充1, 步长1) -> ReLU激活函数 -> 卷积层(3x3卷积核, 填充1, 步长1) ->
# ReLU激活函数 -> 池化层(2x2池化核, 步长2) -> 全连接层(4096个神经元) -> ReLU激活函数 -> 全连接层(4096个神经元) -> ReLU激活函数 -> 输出层
def vggBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return block


class VGG11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG11, self).__init__()
        self.vggblocks = nn.Sequential(
            *[vggBlock(in_channels, out_channels)
            for i in range(n)]
        )
