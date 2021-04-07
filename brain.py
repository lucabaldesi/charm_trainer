import torch
import torch.nn as nn
import torch.nn.functional as F


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        self.chunk_size = chunk_size
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(8, 4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4*self.chunk_size//8, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        y = F.max_pool1d(torch.tanh(self.conv1(x)), 2)
        y = F.max_pool1d(torch.tanh(self.conv2(y)), 2)
        y = F.max_pool1d(torch.tanh(self.conv3(y)), 2)
        y = y.view(-1, 4*self.chunk_size//8)
        y = torch.tanh(self.fc1(y))
        y = self.fc2(y)
        return y
