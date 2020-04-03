import numpy as np
from pathlib import Path
from src.models import train_segmenter, train_classifier


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

    def test_classifier_end_to_end(self, tmp_path):

        self._setup(data_dir=tmp_path)

        train_classifier(max_epochs=1, data_dir=tmp_path)

    def test_segmenter_end_to_end(self, tmp_path):

        self._setup(data_dir=tmp_path)

        train_segmenter(max_epochs=1, data_dir=tmp_path)
