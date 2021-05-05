import torch
import torch.nn as nn
from .fc import FCNet

class options_encoder(nn.Module):
    def __init__(self,dims,act,dropout=0):
        super(options_encoder, self).__init__()
        self.fc = FCNet(dims,act= act,dropout=dropout)

    def forward(self, x):
        out = self.fc(x)
        return out