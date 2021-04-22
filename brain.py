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


class BrainConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.div = self.kernel_size-1
        self.pad = (self.div)//2
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv1d(self.ch_in, self.ch_out, kernel_size=self.kernel_size, padding=self.pad)
        self.batch_norm = nn.BatchNorm1d(track_running_stats=False, num_features=self.ch_out)

    def forward(self, x):
        y = F.max_pool1d(torch.relu(self.batch_norm(self.conv(x))), self.div)
        return y

    def output_n(self, input_n):
        n = k_conv_out_n(1, input_n, self.kernel_size, self.div, self.pad)
        return (n, self.ch_out)


class BrainConvSkip(BrainConv):
    def forward(self, x):
        y = F.max_pool1d(torch.relu(self.batch_norm(self.conv(x))) + x, self.div)
        return y


class BrainLine(nn.Module):
    def __init__(self, inputs, outputs, p):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.line = nn.Linear(inputs, outputs)

    def forward(self, x):
        y = self.dropout(x)
        y = torch.selu(self.line(y))
        return y


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        chs = 64  # convolution output channels
        self.conv_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()

        self.conv_layers.append(BrainConv(2, chs, 11))
        self.conv_layers.append(BrainConv(chs, chs, 3))
        for _ in range(5):
            self.conv_layers.append(BrainConv(chs, chs, 3))

        self.ll1_n = chunk_size
        for c in self.conv_layers:
            self.ll1_n = c.output_n(self.ll1_n)[0]
        self.ll1_n *= chs
        self.ll2_n = 64
        self.ll3_n = 64

        self.line_layers.append(BrainLine(self.ll1_n, self.ll2_n, 0.4))
        self.line_layers.append(BrainLine(self.ll2_n, self.ll3_n, 0.4))
        self.line_layers.append(BrainLine(self.ll3_n, 3, 0.4))

        print(f"Inner nodes: {self.ll1_n}")
        print(f"Parameters: {params_count(self)}")

    def forward(self, x):
        for layer in self.conv_layers[:2]:
            x = layer(x)
        for layer in self.conv_layers[2:]:
            x = layer(x)

        x = x.view(-1, self.ll1_n)
        for layer in self.line_layers:
            x = layer(x)

        return x
