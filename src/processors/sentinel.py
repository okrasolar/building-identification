from pathlib import Path
import xarray as xr

from .base import BaseProcessor
from ..utils import make_sentinel_dataset_name


class SentinelProcessor(BaseProcessor):
    def __init__(self, data_dir: Path, country_code: str) -> None:
        self.country_code: str = country_code
        self.dataset = make_sentinel_dataset_name(country_code)
        super().__init__(data_dir)

    def process_single(self, idx: int, filepath: Path) -> None:
        ds = (
            xr.open_rasterio(filepath)
            .to_dataset(name="sentinel")
            .rename({"x": "lon", "y": "lat"})
        )
        newname = f"data_{idx}.nc"

        ds.to_netcdf(self.processed_dir / newname)

    def process(self, **kwargs) -> None:
        r"""Take the tiff file, save it as a netcdf file and make sure
        the pixels are scaled correctly.

        We don't concatenate datasets, to prevent enormous datasets
        from being generated. We *do* name them a little better
        """

        tif_files = self.raw_data_dir.glob("*")
        for idx, filepath in enumerate(tif_files):
            if filepath.name.endswith(".tif"):
                self.process_single(idx, filepath)
            else:
                print(f"Skipping {filepath.name}")
