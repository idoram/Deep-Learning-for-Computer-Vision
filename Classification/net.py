import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels, num_classes, act):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0, bias=True)

        act = act.lower()
        assert act in {"tanh", "sigmoid", "relu"}, "[ERROR] 只能选择Tanh，Sigmoid或ReLU激活函数，无法使用{}".format(act)
        if act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "relu":
            self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.dense1 = nn.Linear(4 * 4 * 16, 120)
        self.dense2 = nn.Linear(120, 82)
        self.dense3 = nn.Linear(82, num_classes)
        self.act1 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.maxpool(out)
        out = self.act(self.conv2(out))
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.act(self.dense1(out))
        out = self.act(self.dense2(out))
        # out = self.act1(self.dense3(out))
        out = self.dense3(out)
        return out
