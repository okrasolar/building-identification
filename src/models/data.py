import numpy as np
import torch
from pathlib import Path

from torch.utils.data import Dataset

from typing import Tuple


class ClassifierDataset(Dataset):
    def __init__(self, data_dir: Path, train_set: bool = True) -> None:

        features_dir = data_dir / "features"

        if train_set:
            subset_name = "training"
        else:
            subset_name = "validation"

        with_buildings = list(
            (features_dir / "with_buildings" / subset_name).glob("*/x.npy")
        )
        without_buildings = list(
            (features_dir / "without_buildings" / subset_name).glob("*/x.npy")
        )

        self.y = torch.as_tensor(
            [1.0 for _ in with_buildings] + [0.0 for _ in without_buildings]
        )
        self.x_files = with_buildings + without_buildings

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.y[index]

        # pytorch wants channels first, not channels last as they are saved here
        x = np.moveaxis(np.load(self.x_files[index]), -1, 0)
        return torch.as_tensor(x.copy()).float(), y.float()


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: Path, train_set: bool = True) -> None:

        features_dir = data_dir / "features"

        if train_set:
            subset_name = "training"
        else:
            subset_name = "validation"

        self.data_folders = [
            i
            for i in (features_dir / "with_buildings" / subset_name).glob("*")
            if i.is_dir()
        ]

    def __len__(self) -> int:
        return len(self.data_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder = self.data_folders[index]

        x_file = folder / "x.npy"
        y_file = folder / "y.npy"

        # pytorch wants channels first, not channels last as they are saved here
        x = np.moveaxis(np.load(x_file), -1, 0)
        y = np.load(y_file)
        return torch.as_tensor(x.copy()).float(), torch.as_tensor(y.copy()).float()
