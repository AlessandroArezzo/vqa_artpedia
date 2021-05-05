import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch

class AnswerClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(AnswerClassifier, self).__init__()
        layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        logits = self.main(x)
        logits = self.sigmoid(logits)
        return logits
