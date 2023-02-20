import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expension = 1

    def __init__(self, out_channels, stride=1):
        super().__init__()
        nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=(3, 3), padding=1, stride=(stride, stride)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, stride=stride)

        )

    pass




