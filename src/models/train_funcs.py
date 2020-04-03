from pathlib import Path
import pytorch_lightning as pl
from .models import Classifier, Segmenter


def train_classifier(max_epochs: int, data_dir: Path) -> Classifier:

    classifier = Classifier(data_dir=data_dir)

    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(classifier)

    return classifier


def train_segmenter(max_epochs: int, data_dir: Path) -> Segmenter:

    segmenter = Segmenter(data_dir=data_dir)

    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(segmenter)

    return segmenter
