import torch
import pytest

from src.models.segnet.encoder import Encoder


class TestEncoder:
    @pytest.mark.skip(reason="Test too slow")
    def test_encoder(self):

        batch_size, height, width, channels = 100, 224, 224, 3
        input_tensor = torch.ones((batch_size, channels, height, width))

        model = Encoder(pretrained=False)  # To not have to download the model

        with torch.no_grad():
            output, indices = model(input_tensor)

        # checks we have the right number of indices for the decoders
        assert len(indices) == 5
