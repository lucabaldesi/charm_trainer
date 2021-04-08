import torch
import torch.nn as nn
import torch.nn.functional as F


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        self.decimated = 2*chunk_size//(2**3)

        self.mods = nn.ModuleList()
        self.mods.append(nn.Conv1d(2, 2, kernel_size=3, padding=1))
        for _ in range(2):
            self.mods.append(nn.Conv1d(2, 2, kernel_size=3, padding=1))
        self.fc1 = nn.Linear(self.decimated, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        y = x
        for conv in self.mods:
            y = F.max_pool1d(torch.tanh(conv(y)), 2)
        y = y.view(-1, self.decimated)
        y = torch.tanh(self.fc1(y))
        y = self.fc2(y)
        return y
