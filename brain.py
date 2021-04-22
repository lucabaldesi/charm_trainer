import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_n(inputs, kernel_size, padding, stride, dilation=1):
    '''
    Compute the output size of a convolution or pool torch function.
    For convolution, stride should default to 1, for pooling, to kernel_size
    '''
    outs = inputs + 2*padding - dilation*(kernel_size-1) - 1
    outs = outs/stride + 1
    outs = math.floor(outs)
    return int(outs)


def k_conv_out_n(k, inputs, kernel_size, pool_kernel_size, padding):
    '''
    Compute the output size of a series of convolution (and optionally pooling) layers).
    '''
    n = 0
    out = inputs
    for i in range(k):
        out = conv_out_n(out, kernel_size, padding, 1)
        if pool_kernel_size > 0:
            out = conv_out_n(out, pool_kernel_size, 0, pool_kernel_size)
    return out


def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        chs = 32  # convolution output channels

        self.cl0 = nn.Conv1d(2, 2, kernel_size=11, padding=5)
        self.cl1 = nn.Conv1d(2, chs, kernel_size=3, padding=1)
        self.cl2 = nn.Conv1d(chs, chs, kernel_size=3, padding=1)
        self.cl3 = nn.Conv1d(chs, chs, kernel_size=3, padding=1)
        self.cl4 = nn.Conv1d(chs, chs, kernel_size=3, padding=1)
        self.cl5 = nn.Conv1d(chs, chs, kernel_size=3, padding=1)
        self.cl6 = nn.Conv1d(chs, chs, kernel_size=3, padding=1)

        self.bn0 = nn.BatchNorm1d(num_features=2)
        self.bn1 = nn.BatchNorm1d(num_features=chs)
        self.bn2 = nn.BatchNorm1d(num_features=chs)
        self.bn3 = nn.BatchNorm1d(num_features=chs)
        self.bn4 = nn.BatchNorm1d(num_features=chs)
        self.bn5 = nn.BatchNorm1d(num_features=chs)
        self.bn6 = nn.BatchNorm1d(num_features=chs)

        self.ll1_n = k_conv_out_n(1, chunk_size, 11, 10, 5)
        self.ll1_n = k_conv_out_n(6, self.ll1_n, 3, 2, 1)*chs
        self.ll2_n = 12
        self.ll3_n = 12

        self.dpout1 = nn.Dropout(p=0.0)
        self.dpout2 = nn.Dropout(p=0.0)
        self.dpout3 = nn.Dropout(p=0.0)

        self.fc1 = nn.Linear(self.ll1_n, self.ll2_n)
        self.fc2 = nn.Linear(self.ll2_n, self.ll3_n)
        self.fc3 = nn.Linear(self.ll3_n, 3)
        print(f"Inner nodes: {self.ll1_n}")
        print(f"Parameters: {params_count(self)}")

    def forward(self, x):
        y = x
        y = F.max_pool1d(torch.relu((self.cl0(y))), 10)
        y = F.max_pool1d(torch.relu((self.cl1(y))), 2)
        y1 = y
        y = F.max_pool1d(torch.relu((self.cl2(y)))+y1, 2)
        y1 = y
        y = F.max_pool1d(torch.relu((self.cl3(y)))+y1, 2)
        y1 = y
        y = F.max_pool1d(torch.relu((self.cl4(y)))+y1, 2)
        y1 = y
        y = F.max_pool1d(torch.relu((self.cl5(y)))+y1, 2)
        y1 = y
        y = F.max_pool1d(torch.relu((self.cl6(y)))+y1, 2)

        y = y.view(-1, self.ll1_n)
        y = self.dpout1(y)
        y = torch.selu(self.fc1(y))
        y = self.dpout2(y)
        y = torch.selu(self.fc2(y))
        y = self.dpout3(y)
        y = torch.selu(self.fc3(y))

        return y
