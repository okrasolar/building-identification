import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl

from typing import List, Dict, Tuple


class Encoder(pl.LightningModule):

    # The encoder corresponds to the first
    # 13 layers of VGG16

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        vgg = models.vgg16_bn(pretrained=pretrained, progress=True).features

        module_list: List[nn.Module] = []

        for layer_idx in range(len(vgg)):
            layer = vgg[layer_idx]

            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                module_list.append(layer)
            else:
                module_list.append(layer)

        self.encoder = nn.ModuleList(module_list)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:  # type: ignore

        counter = 1
        indices_dict: Dict[int, torch.Tensor] = {}

        for layer in self.encoder:

            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)

                indices_dict[counter] = indices
                counter += 1
            else:
                x = layer(x)

        return x, indices_dict
