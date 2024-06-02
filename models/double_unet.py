import torch
import torch.nn as nn
from torch.nn.functional import pad

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DoubleConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.seq(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0)
        self.helper = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.maxpool(x)
        return self.helper(x)

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.helper = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        x1 = self.upconv(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat((x1, x2), dim=1)
        return self.helper(x)

class DoubleUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2, embedding_size: int = 64, kernel_size: int = 2, stride: int = 2) -> None:
        super(DoubleUNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.encoders = nn.ModuleList()
        self.encoders.append(DoubleConv(in_channels, embedding_size))
        for _ in range(n_encoders - 1):
            self.encoders.append(Encoder(embedding_size, 2 * embedding_size))
            embedding_size *= 2

        self.decoders = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.decoders.append(Decoder(embedding_size, embedding_size // 2))
            embedding_size //= 2
        self.decoders.append(Decoder(embedding_size, out_channels))

    def forward(self, x: torch.tensor) -> torch.tensor:
        residuals = list()
        for encoder in self.encoders:
            x = encoder(x)
            residuals.append(x)

        x = residuals.pop()
        for i in range(len(residuals)):
            x = self.decoders[i](x, residuals[-(i + 1)])

        x = self.maxpool(x)
        return x
