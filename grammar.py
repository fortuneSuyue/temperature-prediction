import os
import random

import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, channels=1, n_kernel=16):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=channels, out_channels=n_kernel, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=n_kernel, out_channels=channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.cnn1(x)
        return self.cnn2(x)


if __name__ == '__main__':
    input_x = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 1, 8))).float()
    model = Net(channels=1, n_kernel=16)
    pred = model(input_x)
    print(pred.shape)
