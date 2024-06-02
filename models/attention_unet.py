import torch
import torch.nn as nn
from .unet import ConvBlock, UpConv

class AttentionBlock(nn.Module):
    def __init__(self, prev_layer: int, encoder_layer: int, coef: int) -> None:
        super(AttentionBlock, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(prev_layer, coef, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(coef)
        )
        self.trans = nn.Sequential(
            nn.Conv2d(encoder_layer, coef, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(coef)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(coef, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prev_gate: torch.Tensor, skip_connection: torch.Tensor) -> torch.tensor:
        g = self.gate(prev_gate)
        x = self.trans(skip_connection)
        psi = self.relu(g + x)
        psi = self.psi(psi)
        return skip_connection * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2, embedding_size: int = 64, kernel_size: int = 2, stride: int = 2) -> None:
        super(AttentionUNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, embedding_size))
        for _ in range(n_encoders - 1):
            self.encoders.append(ConvBlock(embedding_size, 2 * embedding_size))
            embedding_size *= 2

        self.bottleneck = ConvBlock(embedding_size, 2 * embedding_size)

        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.decoders.append(UpConv(embedding_size, embedding_size // 2))
            self.attentions.append(AttentionBlock(embedding_size // 2, embedding_size // 2, embedding_size // 4))
            embedding_size //= 2

        self.final_conv = nn.Conv2d(embedding_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = list()

        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)

        for i, (decoder, attention) in enumerate(zip(self.decoders, self.attentions)):
            x = decoder(x)
            x = attention(x, encoder_outputs[-(i + 1)])
            x = torch.cat((encoder_outputs[-(i + 1)], x), dim=1)

        return self.final_conv(x)