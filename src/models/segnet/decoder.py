import torch
from torch import nn
import pytorch_lightning as pl

from typing import Dict


class Decoder(pl.LightningModule):
    def __init__(self, num_labels: int) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 512, 3),
                DecoderBlock(512, 256, 3),
                DecoderBlock(256, 128, 3),
                DecoderBlock(128, 64, 2),
                DecoderBlock(64, num_labels, 2),
            ]
        )

    def forward(  # type: ignore
        self, x: torch.Tensor, indices_dict: Dict[int, torch.Tensor]
    ) -> torch.Tensor:

        for idx, block in enumerate(self.blocks):
            indices = indices_dict[len(self.blocks) - idx]
            x = block(x, indices)

        return x


class DecoderBlock(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(in_channels, out_channels if i == depth - 1 else in_channels)
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:  # type: ignore

        x = self.unpool(x, indices)
        return self.conv_layers(x)


class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.block(x)
