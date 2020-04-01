import torch
import pytorch_lightning as pl

from .encoder import Encoder
from .decoder import Decoder


class SegNet(pl.LightningModule):
    """
    A segnet, as described in https://arxiv.org/pdf/1511.00561.pdf

    Attributes
    ----------
    num_labels: The number of output labels being predicted
    pretrained: Whether or not to use the pretrained VGG16 model
    """

    def __init__(self, num_labels: int, pretrained: bool = True) -> None:
        super().__init__()

        self.encoder = Encoder(pretrained=pretrained)
        self.decoder = Decoder(num_labels=num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        x, indices_dict = self.encoder(x)
        return self.decoder(x, indices_dict)
