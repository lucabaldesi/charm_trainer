import torch
import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def output2class(model_output, coverage, abstain_class):
    b = F.softmax(model_output, dim=1) * F.log_softmax(model_output, dim=1)
    H = -1.0 * b.sum(dim=1)
    _, predicted = torch.max(model_output, dim=1)
    predicted = (H < coverage).int().mul(predicted)
    predicted += (H >= coverage).int().mul(abstain_class)
    return predicted


