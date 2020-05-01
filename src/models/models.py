from pathlib import Path
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


from .unet import Classifier as UNetClassifier
from .unet import UNet
from .segnet import Encoder, SegNet
from .data import ClassifierDataset, SegmentationDataset

from typing import Any, Dict, Tuple, Type


class Base(pl.LightningModule):

    hparams: Namespace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.binary_cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.binary_cross_entropy(y_hat.squeeze(-1), y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {"val_loss": avg_loss, "log": {"val_loss": avg_loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        default_args: Dict[str, Tuple[Type, Any]] = {
            "--data_dir": (str, str(Path("../data").absolute())),
            "--learning_rate": (float, 0.02),
            "--batch_size": (int, 64),
            "--model": (str, "unet"),
        }

        for key, val in default_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--pretrained", dest="pretrained", action="store_true")
        parser.add_argument("--not_pretrained", dest="pretrained", action="store_false")
        parser.set_defaults(pretrained=True)

        return parser


class Classifier(Base):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        self.hparams = hparams
        self.data_dir = Path(hparams.data_dir)

        if hparams.model == "unet":
            self.model = UNetClassifier(hparams.pretrained)
        elif hparams.model == "segnet":
            self.model = Encoder(hparams.pretrained)
        else:
            raise AssertionError(f"Unknown model {hparams}")

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            ClassifierDataset(data_dir=self.data_dir, train_set=True),
            batch_size=self.hparams.batch_size,
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            ClassifierDataset(data_dir=self.data_dir, train_set=False),
            batch_size=self.hparams.batch_size,
        )


class Segmenter(Base):
    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.data_dir = Path(hparams.data_dir)

        if hparams.model == "unet":
            self.model = UNet(num_labels=1, pretrained=hparams.pretrained)
        elif hparams.model == "segnet":
            self.model = SegNet(num_labels=1, pretrained=hparams.pretrained)
        else:
            raise AssertionError(f"Unknown model {hparams.model}")

    def load_classifier(self, classifier: Classifier) -> None:
        self.model.load_state_dict(classifier.model.state_dict(), strict=False)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            SegmentationDataset(data_dir=self.data_dir, train_set=True),
            batch_size=self.hparams.batch_size,
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            SegmentationDataset(data_dir=self.data_dir, train_set=False),
            batch_size=self.hparams.batch_size,
        )
