import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_n(inputs, kernel_size, padding, dilation=1, stride=1):
    outs = inputs + 2*padding - dilation*(kernel_size-1) - 1
    outs = outs/stride + 1
    outs = math.floor(outs)
    return int(outs)


def k_conv_out_n(k, inputs, kernel_size, padding, dilation=1, stride=1):
    n = 0
    out = inputs
    for i in range(k):
        out = conv_out_n(out, kernel_size, padding, dilation, stride)
        out = math.floor(out/kernel_size)
    return out


def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        chs = 4  # channels
        self.cl1 = nn.Conv1d(2, chs, kernel_size=11)
        self.cl2 = nn.Conv1d(chs, chs, kernel_size=11)
        self.cl3 = nn.Conv1d(chs, chs, kernel_size=11)

        self.ll1_n = k_conv_out_n(3, chunk_size, 11, 0)*chs
        self.ll2_n = 32

        self.fc1 = nn.Linear(self.ll1_n, self.ll2_n)
        self.fc2 = nn.Linear(self.ll2_n, 3)
        print(f"Inner nodes: {self.ll1_n}")
        print(f"Parameters: {params_count(self)}")

    def forward(self, x):
        y = x
        y = F.max_pool1d(torch.tanh(self.cl1(y)), 11)
        y = F.max_pool1d(torch.tanh(self.cl2(y)), 11)
        y = F.max_pool1d(torch.tanh(self.cl3(y)), 11)
        y = y.view(-1, self.ll1_n)
        y = torch.tanh(self.fc1(y))
        y = self.fc2(y)
        return y
