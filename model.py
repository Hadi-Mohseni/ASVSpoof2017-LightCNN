from torch import nn
import torch


class mfm(nn.Module):
    """
    MFM activation function that merges channels to
    keep the model light. Simply divides channels into
    two groups and then selects the maximum.

    Paper: A light CNN for deep face representation with noisy labels
    """

    def __init__(self, out_channels: int, type: int = 1) -> None:
        """
        Parameters
        ----------
        out_channels : int
            number of channels in the output
        type : int, optional
            type=1 --> two group of channels and uses maximum `out.size=(B, out_channels/2, H, W)`
            type=2 --> three groups, uses max and mean `out.size=(B, 2*out_channels/3, H, W)`
            by default 1
        """
        super(mfm, self).__init__()
        self.out_channels = out_channels
        self.type = type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == 1:
            # split channels into two groups
            x = torch.split(x, self.out_channels, 1)
            out = torch.max(x[0], x[1])
        if self.type == 2:
            # split channels into three groups
            x = torch.split(x, int(self.out_channels / 2), 1)
            out1 = torch.max(x[0], x[1])
            out2 = (x[1] + x[2]) / 2
            out = torch.cat([out1, out2], dim=1)

        return out


class NinBlock(nn.Module):
    """
    Network in network block
    ---------

    Network in network block adds a mapping like mlp
    (here a convolution is used) between each two pair of
    convolution layers.

    Paper: Network in network
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of expected channels as output
        """
        super(NinBlock, self).__init__()
        self.nin = nn.Conv2d(in_channels, 2 * in_channels, 1, 1)
        self.mfm1 = mfm(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, 3, 1, padding=1)
        self.mfm2 = mfm(out_channels)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        torch.nn.init.xavier_normal_(self.nin.weight)
        torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nin(x)
        x = self.mfm1(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.mfm2(x)
        x = self.maxpool(x)
        out = self.bn2(x)

        return out


class LightCNN(nn.Module):
    """
    LightCNN
    ----------

    An implementaion of LightCNN used for ASVSpoof2017
    Replay attack. note that this model is being fed with
    spectrogram.

    Paper: Audio replay attack detection with deep learning frameworks
    """

    def __init__(self, in_channels: int = 1) -> None:
        """
        Parameters
        ----------
        in_channels : int
            number of input channels.
        """
        super(LightCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 1, padding=2),
            mfm(16),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )

        torch.nn.init.xavier_normal_(self.block1[0].weight)

        self.backbone = nn.Sequential(
            NinBlock(16, 24),
            NinBlock(24, 32),
            NinBlock(32, 16),
            NinBlock(16, 16),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184, 64),
            mfm(32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.7),
            nn.Linear(32, 2),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.backbone(x)
        x = self.classifier(x)
        pred = self.softmax(x)
        return pred
