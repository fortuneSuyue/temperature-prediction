from torch import nn
import torch


class LogCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        loss = torch.log(torch.cosh(pred - true))
        return torch.sum(loss)
