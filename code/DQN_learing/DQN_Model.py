import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):

    def __init__(self, dim):
        super(DQN, self).__init__()
        l1 = 64
        self.w1 = nn.Linear(dim, l1)
        # self.w1.cuda()# 基于CUDA进行GPU并行加速
        self.w2 = nn.Linear(l1, 1)
        # self.w2.cuda()

    def forward(self, input):
        out = self.w1(input)#.cuda()
        out = F.relu_(out)#.cuda()

        out = self.w2(out)#.cuda()

        return out

