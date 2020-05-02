from argparse import Namespace
import numpy as np
from pathlib import Path
from src.models import train_model, Classifier, Segmenter


class TestModelsEndToEnd:
    def _setup(self, data_dir: Path, num_examples: int = 3):
        height, width, channels = 224, 224, 3
        input_x = np.ones((height, width, channels))
        input_y = np.ones((height, width))

        train_save_dir = data_dir / "features" / "with_buildings" / "training"
        val_save_dir = data_dir / "features" / "with_buildings" / "validation"

        train_save_dir.mkdir(parents=True)
        val_save_dir.mkdir(parents=True)

        for i in range(num_examples):

            folder_train_dir = train_save_dir / str(i)
            folder_val_dir = val_save_dir / str(i)

            folder_train_dir.mkdir(parents=True)
            folder_val_dir.mkdir(parents=True)

            np.save(folder_train_dir / "x.npy", input_x)
            np.save(folder_val_dir / "x.npy", input_x)

            np.save(folder_train_dir / "y.npy", input_y)
            np.save(folder_val_dir / "y.npy", input_y)

    @staticmethod
    def _get_args(tmp_path: Path) -> Namespace:
        args = {
            "data_dir": str(tmp_path.absolute()),
            "model": "unet",
            "learning_rate": 0.02,
            "batch_size": 64,
            "pretrained": False,
            "max_epochs": 1,
            "patience": 1,
        }

        return Namespace(**args)

    def test_classifier_end_to_end(self, tmp_path):

        self._setup(data_dir=tmp_path)
        hparams = self._get_args(tmp_path)
        model = Classifier(hparams)
        train_model(model, hparams)

    def test_segmenter_end_to_end(self, tmp_path):

        self._setup(data_dir=tmp_path)
        hparams = self._get_args(tmp_path)
        model = Segmenter(hparams)
        train_model(model, hparams)
