import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 8, embedding_size: int = 64, kernel_size: int = 2, stride: int = 2) -> None:
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, embedding_size))
        for _ in range(n_encoders - 2):
            self.encoders.append(ConvBlock(embedding_size, 2 * embedding_size))
            embedding_size *= 2
        
        self.bottleneck = ConvBlock(embedding_size, 2 * embedding_size)
        embedding_size *= 2

        self.upconvs = nn.ModuleList()
        self.convblocks = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.upconvs.append(UpConv(embedding_size, embedding_size // 2))
            self.convblocks.append(ConvBlock(embedding_size, embedding_size // 2))
            embedding_size //= 2

        self.final_conv = nn.Conv2d(embedding_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = list()

        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)

        for i, (upconv, convblock) in enumerate(zip(self.upconvs, self.convblocks)):
            x = upconv(x)
            enc_out = encoder_outputs[-(i + 1)]

            # Ensure dimensions match by cropping
            if x.shape[2:] != enc_out.shape[2:]:
                diffY = enc_out.size(2) - x.size(2)
                diffX = enc_out.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat((enc_out, x), dim=1)
            x = convblock(x)

        return self.final_conv(x)
