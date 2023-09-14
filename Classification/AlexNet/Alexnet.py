import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        # 假设输入一个形状为（1, 1, 227, 227)的图像数据
        super(AlexNet, self).__init__()
        # (1, 1, 227, 227) -> (1, 96, 55, 55)
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=11, stride=4)
        self.act = nn.ReLU()
        # (1, 96, 55, 55) -> (1, 96, 27, 27)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        # (1, 96, 27, 27) -> (1, 256, 27, 27)
        # (1, 256, 27, 27) -> (1, 256, 13, 13) MaxPool
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding="same")
        # (1, 256, 13, 13) -> (1, 384, 13, 13)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding="same")
        # (1, 384, 13, 13) -> (1, 384, 13, 13)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding="same")
        # (1, 384, 13, 13) -> (1, 256, 13, 13)
        # (1, 256, 13, 13) -> (1, 256, 6, 6) MaxPool
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding="same")
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 6 * 6, 4096)
        self.drop = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.maxp(self.act(self.conv1(x)))
        out = self.maxp(self.act(self.conv2(out)))
        out = self.act(self.conv3(out))
        out = self.act(self.conv4(out))
        out = self.maxp(self.act(self.conv5(out)))
        out = self.flatten(out)
        out = self.drop(self.act(self.dense1(out)))
        out = self.drop(self.act(self.dense2(out)))
        out = self.softmax(self.dense3(out))

        return out
