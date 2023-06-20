from torch import nn
import torch


class mfm(nn.Module):
    def __init__(self, out_channels: int, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        self.type = type

    def forward(self, x):
        if self.type == 1:
            x = torch.split(x, self.out_channels, 1)
            out = torch.max(x[0], x[1])
        else:
            x = torch.split(x, int(self.out_channels / 2), 1)
            out1 = torch.max(x[0], x[1])
            out2 = (x[1] + x[2]) / 2
            out = torch.cat([out1, out2], dim=1)

        return out


class LightCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(LightCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, 1)
        self.mfm1 = mfm(16)

        self.conv2a = nn.Conv2d(16, 32, 1, 1)
        self.mfm2a = mfm(16)
        self.conv2b = nn.Conv2d(16, 48, 3, 1)
        self.mfm2b = mfm(24)

        self.conv3a = nn.Conv2d(24, 48, 1, 1)
        self.mfm3a = mfm(32, type=2)
        self.conv3b = nn.Conv2d(32, 64, 3, 1)
        self.mfm3b = mfm(32)

        self.conv4a = nn.Conv2d(32, 64, 1, 1)
        self.mfm4a = mfm(32)
        self.conv4b = nn.Conv2d(32, 32, 3, 1)
        self.mfm4b = mfm(16)

        self.conv5a = nn.Conv2d(16, 32, 1, 1)
        self.mfm5a = mfm(16)
        self.conv5b = nn.Conv2d(16, 32, 3, 1)
        self.mfm5b = mfm(16)

        self.flat = nn.Flatten()

        self.fc6 = nn.Linear(16, 2)
        self.mfm6 = mfm(1)

        self.fc7 = nn.Linear(1, 2)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.maxpool(x)
        x = self.mfm2a(self.conv2a(x))
        x = self.mfm2b(self.conv2b(x))
        x = self.maxpool(x)
        x = self.mfm3a(self.conv3a(x))
        x = self.mfm3b(self.conv3b(x))
        x = self.maxpool(x)
        x = self.mfm4a(self.conv4a(x))
        x = self.mfm4b(self.conv4b(x))
        x = self.maxpool(x)
        x = self.mfm5a(self.conv5a(x))
        x = self.mfm5b(self.conv5b(x))
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.mfm6(self.fc6(x))
        x = self.fc7(x)
        return x
