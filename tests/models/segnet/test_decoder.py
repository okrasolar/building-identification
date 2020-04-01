import torch
import pytest

from src.models.segnet.encoder import Encoder
from src.models.segnet.decoder import Decoder


class TestDecoder:
    @pytest.mark.skip(reason="Test too slow")
    def test_decoder(self):

        batch_size, height, width, channels, num_labels = 100, 224, 224, 3, 10
        input_tensor = torch.ones((batch_size, channels, height, width))

        encoder = Encoder(pretrained=False)  # To not have to download the model
        decoder = Decoder(num_labels=num_labels)

        with torch.no_grad():
            interim_output, indices = encoder(input_tensor)
            output = decoder(interim_output, indices)

        assert output.shape == (batch_size, num_labels, height, width)
