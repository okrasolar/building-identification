import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


from .unet import Classifier as UNetClassifier
from .segnet import Encoder
from .data import ClassifierDataset


class Classifier(pl.LightningModule):
    def __init__(self, data_dir: Path, model: str = "unet", pretrained: bool = True):
        super().__init__()

        self.data_dir = data_dir

        if model == "unet":
            self.model = UNetClassifier(pretrained)
        elif model == "segnet":
            self.model = Encoder(pretrained)
        else:
            raise AssertionError(f"Unknown model {model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.binary_cross_entropy(y_hat, y.unsqueeze(-1))}

    def train_dataloader(self):
        return DataLoader(ClassifierDataset(data_dir=self.data_dir), batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
