import torch
import pytest
from src.models.segnet import SegNet


class TestSegnet:
    @pytest.mark.skip(reason="Test too slow")
    def test_segnet(self):

        batch_size, height, width, channels, num_labels = 100, 224, 224, 3, 10
        input_tensor = torch.ones((batch_size, channels, height, width))

        model = SegNet(
            pretrained=False, num_labels=num_labels
        )  # To not have to download the model

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, num_labels, height, width)
