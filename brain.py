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
        chs = 10  # convolution output channels

        self.cl1 = nn.Conv1d(2, chs, kernel_size=11, padding=5)
        self.cl2 = nn.Conv1d(chs, chs, kernel_size=11, padding=5)
        self.cl3 = nn.Conv1d(chs, chs, kernel_size=11, padding=5)

        self.ll1_n = k_conv_out_n(3, chunk_size, 11, 10, 5)*chs
        self.ll2_n = 10

        self.fc1 = nn.Linear(self.ll1_n, self.ll2_n)
        self.fc2 = nn.Linear(self.ll2_n, 3)
        print(f"Inner nodes: {self.ll1_n}")
        print(f"Parameters: {params_count(self)}")

    def forward(self, x):
        y = x
        y = F.max_pool1d(torch.tanh(self.cl1(y)), 10)
        y = F.max_pool1d(torch.tanh(self.cl2(y)), 10)
        y1 = y
        y = F.max_pool1d(torch.tanh(self.cl3(y)) + y1, 10)
        y = y.view(-1, self.ll1_n)
        y = torch.tanh(self.fc1(y))
        y = self.fc2(y)
        return y
