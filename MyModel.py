import torch
import torch.nn as nn
import ResNetRadio
import complexPyTorch
import complexPyTorch.complexLayers

class base_model(nn.Module):
    """
            xc = torch.zeros((x.shape[0], 1, 1, 128), dtype=torch.complex64)
            # x.shape --[b, 3, 224, 224]  y.shape [b]
            y = torch.as_tensor(y, dtype=torch.long)
            xc.real = x[:,0,:].unsqueeze(dim=1).unsqueeze(dim=1)
            xc.imag = x[:,1,:].unsqueeze(dim=1).unsqueeze(dim=1)
    参数说明:

    """
    def __init__(self, args, mode=True):
        super(base_model, self).__init__()
        self.c1 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=1, out_channels=64, kernel_size=(3, 1),
                                                   stride=1, padding='same')
        self.r1 = complexPyTorch.complexLayers.ComplexReLU()
        self.n1 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p1 = nn.MaxPool1d(kernel_size=2)

        self.c2 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                   stride=1, padding='same')
        self.r2 = complexPyTorch.complexLayers.ComplexReLU()
        self.n2 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p2 = nn.MaxPool1d(kernel_size=2)

        self.c3 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r3 = complexPyTorch.complexLayers.ComplexReLU()
        self.n3 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p3 = nn.MaxPool1d(kernel_size=2)

        self.c4 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r4 = complexPyTorch.complexLayers.ComplexReLU()
        self.n4 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p4 = nn.MaxPool1d(kernel_size=2)

        self.c5 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r5 = complexPyTorch.complexLayers.ComplexReLU()
        self.n5 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p5 = nn.MaxPool1d(kernel_size=2)

        self.c6 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r6 = complexPyTorch.complexLayers.ComplexReLU()
        self.n6 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p6 = nn.MaxPool1d(kernel_size=2)

        self.c7 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r7 = complexPyTorch.complexLayers.ComplexReLU()
        self.n7 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p7 = nn.MaxPool1d(kernel_size=2)

        self.c8 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r8 = complexPyTorch.complexLayers.ComplexReLU()
        self.n8 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p8 = nn.MaxPool1d(kernel_size=2)

        self.c9 = complexPyTorch.complexLayers.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1),
                                                             stride=1, padding='same')
        self.r9 = complexPyTorch.complexLayers.ComplexReLU()
        self.n9 = complexPyTorch.complexLayers.BatchNorm1d(128)
        self.p9 = nn.MaxPool1d(kernel_size=2)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9, 1024),
        )
        self.mode = mode
        self.classifier = nn.Linear(1024, args.train_val_dataset_num)

    def forward(self, x):
        # print(x.shape)  # torch.Size([128, 4800, 2])
        x_i = x[:, :, 0]
        x_q = x[:, :, 1]
        x = x_i+1j*x_q
        # print(x.shape)    # torch.Size([128, 4800])
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 3)

        x = self.c1(x)
        x = self.r1(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n1(x)
        x = self.p1(x)
        # print("x1:", x.shape)
        x = x[:, :64, :] + 1j*x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c2(x)
        x = self.r2(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n2(x)
        x = self.p2(x)
        # print("x2:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c3(x)
        x = self.r3(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n3(x)
        x = self.p3(x)
        # print("x3:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c4(x)
        x = self.r4(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n4(x)
        x = self.p4(x)
        # print("x4:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c5(x)
        x = self.r5(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n5(x)
        x = self.p5(x)
        # print("x5:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c6(x)
        x = self.r6(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n6(x)
        x = self.p6(x)
        # print("x6:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c7(x)
        x = self.r7(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n7(x)
        x = self.p7(x)
        # print("x7:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c8(x)
        x = self.r8(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n8(x)
        x = self.p8(x)
        # print("x8:", x.shape)
        x = x[:, :64, :] + 1j * x[:, 64:, :]
        x = torch.unsqueeze(x, 3)
        x = self.c9(x)
        x = self.r9(x)
        x = torch.squeeze(x)
        x = torch.cat([x.real, x.imag], 1)
        x = self.n9(x)
        x = self.p9(x)
        # print("x9:", x.shape)
        x = self.dense(x)
        if self.mode:
            x = self.classifier(x)
        return x


class ResNetForClass(nn.Module):
    def __init__(self, resnet, args, mode=True):
        super(ResNetForClass, self).__init__()
        self.mode = mode

        self.encoder = resnet
        self.classifier = nn.Linear(1024, args.train_val_dataset_num)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 1)
        # torch.size(n, 1, 2, 4800)
        x = self.encoder(x)
        if self.mode:
            x = self.classifier(x)

        return x


class LinearRegression(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = self.linear(x)
        return x


