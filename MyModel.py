import torch
import torch.nn as nn
import ResNetRadio


class ResNetForClass(nn.Module):
    def __init__(self, resnet, args):
        super(ResNetForClass, self).__init__()
        self.encoder = resnet
        self.linear = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, args.train_val_dataset_num)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 1)
        # torch.size(n, 1, 2, 4800)
        x = self.encoder(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

