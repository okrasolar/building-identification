from pathlib import Path
import xarray as xr
import rasterio
from rasterio.mask import mask
from shapely import geometry
import numpy as np

from .base import BaseProcessor
from ..utils import make_hrsl_dataset_name, make_sentinel_dataset_name

from typing import List, Tuple, Optional


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

    def write_temp_hrsl_grid(self, reference_grid: xr.Dataset, filepath: Path) -> None:

        img = rasterio.open(
            self.raw_data_dir / f"hrsl_{self.country_code}_settlement.tif"
        )

        grid_box = geometry.box(
            reference_grid.lon.min(),
            reference_grid.lat.min(),
            reference_grid.lon.max(),
            reference_grid.lat.max(),
        )
        crop, cropTransform = mask(img, [grid_box], crop=True)

        new_metadata = img.meta
        new_metadata.update(
            {
                "driver": "GTiff",
                "height": crop.shape[1],
                "width": crop.shape[2],
                "transform": cropTransform,
                "crs": img.crs,
            }
        )

        with rasterio.open(filepath, "w", **new_metadata) as dest:
            dest.write(crop)

    def process(self, **kwargs) -> None:

        sentinel_grids = self.load_sentinel_grids()

        interim_filepath = self.processed_dir / "interim.tif"

        for reference_grid, grid_name in sentinel_grids:
            if kwargs.get("checkpoint", True):
                if (self.processed_dir / grid_name).exists():
                    print("File already processed! Skipping")
                    continue

            if interim_filepath.exists():
                interim_filepath.unlink()

            print(f"Regridding onto {grid_name}")

            self.write_temp_hrsl_grid(reference_grid, filepath=interim_filepath)

            da = xr.open_rasterio(interim_filepath).isel(band=0)
            da = self.zero_array(da)
            if da is not None:
                da = (
                    da.to_dataset(name="hrsl")
                    .rename({"x": "lon", "y": "lat"})
                    .drop("band")
                )

                regridded = self.regrid(ds=da, reference_ds=reference_grid)
                regridded.to_netcdf(self.processed_dir / grid_name)
        interim_filepath.unlink()

    @staticmethod
    def zero_array(da: xr.DataArray) -> Optional[xr.DataArray]:
        """
        The DataArrays should have only values 1, 0. Instead, the 0 is replaced
        with some other *variable* value. This function figures out what that value
        is and replaces it with 0
        """

        unique_values = np.unique(da.values)

        if not ((len(unique_values) == 2) and (1 in unique_values)):
            # This can happen if we have a tile which is only ocean
            print("File only contains one value - skipping!")

        zero_value = unique_values[unique_values != 1][0]
        da.values[da.values == zero_value] = 0

        return da
