from abc import ABC, abstractmethod
from pathlib import Path
import xarray as xr

from ..utils import load_sentinel

from typing import cast


class BaseProcessor(ABC):

    dataset: str

    def __init__(self, data_dir: Path) -> None:

        self.data_dir = data_dir

        self.raw_data_dir = self.data_dir / f"raw/{self.dataset}"
        assert self.raw_data_dir.exists(), "No raw data dir! Has it been exported?"

        self.processed_dir = self.data_dir / f"processed/{self.dataset}"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def process(self, **kwargs) -> None:
        raise NotImplementedError

    def regrid(
        self, ds: xr.Dataset, reference_ds: xr.Dataset, method: str = "nearest"
    ) -> xr.Dataset:
        return ds.reindex(
            {"lat": reference_ds.lat, "lon": reference_ds.lon}, method=method
        )

    @staticmethod
    def load_reference_grid(path_to_grid: Path) -> xr.Dataset:
        r"""Since the regridder only needs to the lat and lon values,
        there is no need to pass around an enormous grid for the regridding.
        In fact, only the latitude and longitude values are necessary!
        """
        full_dataset = load_sentinel(path_to_grid)

        assert {"lat", "lon"} <= set(
            full_dataset.dims
        ), "Dimensions named lat and lon must be in the reference grid"
        return cast(xr.Dataset, full_dataset[["lat", "lon"]])
