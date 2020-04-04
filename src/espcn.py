import numpy as np
import torch
from torch import nn


def PS(T, r):
    T = np.transpose(T, (1, 2, 0))
    rW = r * len(T)
    rH = r * len(T[0])
    C = len(T[0][0]) / (r * r)

    # make sure C is an integer and cast if this is the case
    assert (C == int(C))
    C = int(C)

    res = np.zeros((rW, rH, C))

    for x in range(len(res)):
        for y in range(len(res[x])):
            for c in range(len(res[x][y])):
                res[x][y][c] = \
                    T[x // r][y // r][C * r * (y % r) + C * (x % r) + c]
    return res


def PS_inv(img, r):
    r2 = r * r
    W = len(img) / r
    H = len(img[0]) / r
    C = len(img[0][0])
    Cr2 = C * r2

    # Make sure H and W are integers
    assert (int(H) == H and int(W) == W)
    H, W = int(H), int(W)

    res = np.zeros((W, H, Cr2))

    for x in range(len(img)):
        for y in range(len(img[x])):
            for c in range(len(img[x][y])):
                res[x // r][y // r][C * r * (y % r) + C * (x % r) + c] = img[x][y][c]
    return res


class Net(nn.Module):
    def __init__(self, r, C):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, r * r * C, 3, padding=1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.conv3(x)
        return x