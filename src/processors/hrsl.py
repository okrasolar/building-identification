from pathlib import Path
import xarray as xr
import numpy as np

from .base import BaseProcessor
from ..utils import make_hrsl_dataset_name, make_sentinel_dataset_name

from typing import List, Tuple


class HRSLProcessor(BaseProcessor):
    def __init__(self, data_dir: Path, country_code: str) -> None:
        self.country_code: str = country_code
        self.dataset = make_hrsl_dataset_name(country_code)
        super().__init__(data_dir)

    def load_sentinel_grids(self) -> List[Tuple[xr.Dataset, str]]:
        dataset_name = make_sentinel_dataset_name(self.country_code)
        processed_dataset_dir = self.data_dir / f"raw/{dataset_name}"

        output: List[Tuple[xr.Dataset, str]] = []
        for filepath in processed_dataset_dir.glob("*"):
            if filepath.name.endswith(".tif"):
                output.append((self.load_reference_grid(filepath), filepath.name))
        return output

    def process(self, **kwargs) -> None:

        da = xr.open_rasterio(
            self.raw_data_dir / f"hrsl_{self.country_code}_settlement.tif"
        ).isel(band=0)

        da = (
            self.zero_array(da)
            .to_dataset(name="hrsl")
            .rename({"x": "lon", "y": "lat"})
            .drop("band")
        )

        da.to_netcdf(self.processed_dir / "data.nc")

        sentinel_grids = self.load_sentinel_grids()

        for reference_grid, grid_name in sentinel_grids:
            print(f"Regridding onto {grid_name}")
            regridded = self.regrid(ds=da, reference_ds=reference_grid)
            regridded.to_netcdf(self.processed_dir / grid_name)

    @staticmethod
    def zero_array(da: xr.DataArray) -> xr.DataArray:
        """
        The DataArrays should have only values 1, 0. Instead, the 0 is replaced
        with some other *variable* value. This function figures out what that value
        is and replaces it with 0
        """

        unique_values = np.unique(da.values)

        assert (len(unique_values) == 2) and (1 in unique_values)

        zero_value = unique_values[unique_values != 1][0]
        da.values[da.values == zero_value] = 0

        return da
