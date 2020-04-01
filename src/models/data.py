import numpy as np
import torch
from pathlib import Path

from typing import Optional, List, Tuple


class ClassifierDataset:
    def __init__(self, data_dir: Path, mask: Optional[List[bool]] = None) -> None:

        features_dir = data_dir / "features"

        with_buildings = list((features_dir / "with_buildings").glob("*/x.npy"))
        without_buildings = list((features_dir / "without_buildings").glob("*/x.npy"))

        self.y = torch.as_tensor(
            [1.0 for _ in with_buildings] + [0.0 for _ in without_buildings]
        )
        self.x_files = with_buildings + without_buildings

        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask: List[bool]) -> None:
        """Add a mask to the data
        """
        assert len(mask) == len(
            self.x_files
        ), f"Mask is the wrong size! Expected {len(self.x_files)}, got {len(mask)}"
        self.y = torch.as_tensor(self.y.numpy()[mask])
        self.x_files = [x for include, x in zip(mask, self.x_files) if include]

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.y[index]

        # pytorch wants channels first, not channels last as they are saved here
        x = np.moveaxis(np.load(self.x_files[index]), -1, 0)
        return torch.as_tensor(x.copy()).float(), y.float()
