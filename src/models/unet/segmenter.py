import torch
from torch import nn
import pytorch_lightning as pl

from typing import List

from .base import ResnetBase


class UNet(ResnetBase):
    """A ResNet34 U-Net model, as described in
    https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet-lrg.ipynb
    Attributes:
        imagenet_base: boolean, default: False
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, num_labels: int, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)

        self.target_modules = [str(x) for x in [2, 4, 5, 6]]
        self.hooks = self.add_hooks()

        self.relu = nn.ReLU()
        self.upsamples = nn.ModuleList(
            [
                UpBlock(512, 256, 256),
                UpBlock(256, 128, 256),
                UpBlock(256, 64, 256),
                UpBlock(256, 64, 256),
                UpBlock(256, 3, 16),
            ]
        )
        self.conv_transpose = nn.ConvTranspose2d(16, num_labels, 1)

    def add_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                hooks.append(child.register_forward_hook(self.save_output))
        return hooks

    def retrieve_hooked_outputs(self):
        # to be called in the forward pass, this method returns the tensors
        # which were saved by the forward hooks
        outputs = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                outputs.append(child.output)
        return outputs

    def cleanup(self) -> None:
        # removes the hooks, and the tensors which were added
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                # allows the method to be safely called even if
                # the hooks aren't there
                try:
                    del child.output
                except AttributeError:
                    continue
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def save_output(module, input, output):
        # the hook to add to the target modules
        module.output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_input = x
        x = self.relu(self.pretrained(x))
        # we reverse the outputs so that the smallest output
        # is the first one we get, and the largest the last
        interim = self.retrieve_hooked_outputs()[::-1]

        for upsampler, interim_output in zip(self.upsamples[:-1], interim):
            x = upsampler(x, interim_output)
        x = self.upsamples[-1](x, org_input)
        return self.conv_transpose(x)


class UpBlock(pl.LightningModule):
    def __init__(
        self, in_channels: int, across_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        up_out = across_out = out_channels // 2
        self.conv_across = nn.Conv2d(across_channels, across_out, 1)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, up_out, 2, stride=2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_up, x_across):
        joint = torch.cat(
            (self.conv_transpose(x_up), self.conv_across(x_across)), dim=1
        )
        return self.batchnorm(self.relu(joint))
