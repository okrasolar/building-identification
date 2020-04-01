from torch import nn
from torchvision.models import resnet34
import pytorch_lightning as pl


class ResnetBase(pl.LightningModule):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model
    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        resnet = resnet34(pretrained=pretrained).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError
