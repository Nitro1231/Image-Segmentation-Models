import torch
import torch.nn as nn
from .unet import UpConv

class RecurrentBlock(nn.Module):
    def __init__(self, out_channels: int, t: int = 2) -> None:
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class ResidualRecurrentCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t: int = 2) -> None:
        super(ResidualRecurrentCNNBlock, self).__init__()
        self.residual_recurrent_cnn = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t)
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        x1 = self.residual_recurrent_cnn(x)
        return x + x1

class R2UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2, embedding_size: int = 64, kernel_size: int = 2, stride: int = 2) -> None:
        super(R2UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.encoders = nn.ModuleList()
        self.encoders.append(ResidualRecurrentCNNBlock(in_channels, embedding_size))
        for _ in range(n_encoders - 1):
            self.encoders.append(ResidualRecurrentCNNBlock(embedding_size, 2 * embedding_size))
            embedding_size *= 2

        self.bottleneck = ResidualRecurrentCNNBlock(embedding_size, 2 * embedding_size)

        self.decoders = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.decoders.append(UpConv(embedding_size, embedding_size // 2))
            embedding_size //= 2
        self.decoders.append(UpConv(embedding_size * 2, embedding_size))

        self.final_conv = nn.Conv2d(embedding_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = list()

        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            x = torch.cat((encoder_outputs[-(i + 1)], x), dim=1)

        return self.final_conv(x)
