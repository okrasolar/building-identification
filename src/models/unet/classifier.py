import torch
from torch import nn

from .base import ResnetBase


class Classifier(ResnetBase):
    """A ResNet34 Model

    :param pretrained: Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))
