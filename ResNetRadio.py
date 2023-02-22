import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expension = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), stride=(1, stride)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expension, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels * BasicBlock.expension)

        )
        self.shortcut = nn.Sequential()
        # make x match the output

        if stride != 1 or in_channels != BasicBlock.expension * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expension, kernel_size=(1, 1), stride=(1, stride)),
                nn.BatchNorm2d(out_channels*BasicBlock.expension)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.shortcut(x) + self.residual_function(x))


class BottleNeck(nn.Module):
    expension = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), stride=(1, stride)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expension, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels*BottleNeck.expension)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*BottleNeck.expension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expension, kernel_size=(1, 1), stride=(1, stride)),
                nn.BatchNorm2d(out_channels*BottleNeck.expension)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.shortcut(x)+self.residual_function(x))


class ResNet(nn.Module):
    def __init__(self, block, block_nums):
        super(ResNet, self).__init__()
        self.input_channels = 32
        self.c1 = nn.Sequential(
            nn.Conv2d(1, self.input_channels, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )
        self.c2 = self._make_layer(block, self.input_channels, block_nums[0], stride=2)
        self.c3 = self._make_layer(block, self.input_channels, block_nums[1], stride=2)
        self.c4 = self._make_layer(block, self.input_channels, block_nums[2], stride=2)
        self.c5 = self._make_layer(block, self.input_channels, block_nums[3], stride=2)
        self.aver_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.linear = nn.Linear(self.input_channels * block.expension*2*150, 1024)

    def _make_layer(self, block, output_channels, block_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        layer = []
        for stride in strides:
            layer.append(block(self.input_channels, output_channels, stride))
            self.input_channels = output_channels * block.expension

        return nn.Sequential(*layer)

    def forward(self, x):

        # torch.size(n, 1, 2, 4800)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.aver_pool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)

        return x8


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    resnet50 = resnet50()
    print(resnet50)
    '''    
    resnet18 = resnet18()
    print(resnet18)
    '''


