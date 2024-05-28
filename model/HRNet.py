import torch
import torch.nn as nn

class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()

    def forward(self, input): 
        return input