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
    def __init__(self, in_channels: int, initializer=nn.init.xavier_normal_) -> None:
        super(LightCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, 1, padding=2)
        self.mfm1 = mfm(16)
        initializer(self.conv1.weight)

        self.conv2a = nn.Conv2d(16, 32, 1, 1)
        self.mfm2a = mfm(16)
        self.conv2b = nn.Conv2d(16, 48, 3, 1, padding=1)
        self.mfm2b = mfm(24)
        initializer(self.conv2a.weight)
        initializer(self.conv2b.weight)

        self.conv3a = nn.Conv2d(24, 48, 1, 1)
        self.mfm3a = mfm(32, type=2)
        self.conv3b = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.mfm3b = mfm(32)
        initializer(self.conv3a.weight)
        initializer(self.conv3b.weight)

        self.conv4a = nn.Conv2d(32, 64, 1, 1)
        self.mfm4a = mfm(32)
        self.conv4b = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.mfm4b = mfm(16)
        initializer(self.conv4b.weight)
        initializer(self.conv4a.weight)

        self.conv5a = nn.Conv2d(16, 32, 1, 1)
        self.mfm5a = mfm(16)
        self.conv5b = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.mfm5b = mfm(16)
        initializer(self.conv5b.weight)
        initializer(self.conv5a.weight)

        self.flat = nn.Flatten()

        self.fc6 = nn.Linear(5184, 64)
        self.mfm6 = mfm(32)

        self.fc7 = nn.Linear(32, 2)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.7)
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(24)
        self.bn4 = nn.BatchNorm2d(24)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(16)
        self.bn8 = nn.BatchNorm2d(16)
        self.bn9 = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x = self.conv2a(x)
        x = self.mfm2a(x)
        x = self.bn2(x)

        x = self.conv2b(x)
        x = self.mfm2b(x)
        x = self.maxpool(x)
        x = self.bn3(x)

        x = self.conv3a(x)
        x = self.mfm3a(x)
        x = self.bn4(x)

        x = self.conv3b(x)
        x = self.mfm3b(x)
        x = self.maxpool(x)
        x = self.bn5(x)

        x = self.conv4a(x)
        x = self.mfm4a(x)
        x = self.bn6(x)

        x = self.conv4b(x)
        x = self.mfm4b(x)
        x = self.maxpool(x)
        x = self.bn7(x)

        x = self.conv5a(x)
        x = self.mfm5a(x)
        x = self.bn8(x)

        x = self.conv5b(x)
        x = self.mfm5b(x)
        x = self.maxpool(x)
        x = self.bn9(x)

        x = self.flat(x)
        x = self.fc6(x)
        x = self.mfm6(x)

        x = self.dropout(x)
        x = self.fc7(x)
        x = self.softmax(x)
        return x
