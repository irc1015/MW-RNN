import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        self.norm = nn.GroupNorm(num_groups=2, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        y = self.conv(x)
        y = self.act(self.norm(y))
        return y