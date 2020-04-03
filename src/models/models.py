import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


from .unet import Classifier as UNetClassifier
from .unet import UNet
from .segnet import Encoder, SegNet
from .data import ClassifierDataset, SegmentationDataset


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
        return {"loss": F.binary_cross_entropy(y_hat.squeeze(-1), y)}

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            ClassifierDataset(data_dir=self.data_dir, train_set=True), batch_size=64
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            ClassifierDataset(data_dir=self.data_dir, train_set=False), batch_size=64
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.binary_cross_entropy(y_hat.squeeze(-1), y)}


class Segmenter(pl.LightningModule):
    def __init__(self, data_dir: Path, model: str = "unet", pretrained: bool = True):
        super().__init__()

        self.data_dir = data_dir

        if model == "unet":
            self.model = UNet(num_labels=1, pretrained=pretrained)
        elif model == "segnet":
            self.model = SegNet(num_labels=1, pretrained=pretrained)
        else:
            raise AssertionError(f"Unknown model {model}")

    def load_classifier(self, classifier: Classifier) -> None:
        self.model.load_state_dict(classifier.model.state_dict(), strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.binary_cross_entropy(y_hat.squeeze(1), y)}

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            SegmentationDataset(data_dir=self.data_dir, train_set=True), batch_size=64
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            SegmentationDataset(data_dir=self.data_dir, train_set=False), batch_size=64
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.binary_cross_entropy(y_hat.squeeze(1), y)}
