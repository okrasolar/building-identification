from argparse import Namespace

import pytorch_lightning as pl


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:

    trainer = pl.Trainer(
        default_save_path=hparams.data_folder,
    )
    trainer.fit(model)

    return model
