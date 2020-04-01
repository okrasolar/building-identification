import torch
import pytest
from src.models.unet import UNet


class TestUNet:
    @pytest.mark.skip(reason="Test too slow")
    def test_unet(self):

        batch_size, height, width, channels, num_labels = 100, 224, 224, 3, 10
        input_tensor = torch.ones((batch_size, channels, height, width))

        model = UNet(
            pretrained=False, num_labels=num_labels
        )  # To not have to download the model

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, num_labels, height, width)
