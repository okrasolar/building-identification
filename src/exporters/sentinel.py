from pathlib import Path
from dataclasses import dataclass
from datetime import date
import xarray as xr

import ee

from .base import BaseExporter
from ..utils import make_sentinel_dataset_name, make_hrsl_dataset_name
from . import cloudfree

from typing import List, Tuple

try:
    ee.Initialize()
except Exception:
    print(
        "This code doesn't work unless you have authenticated your earthengine account"
    )


@dataclass
class BoundingBox:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def to_polygon(self) -> ee.Geometry.Polygon:
        return ee.Geometry.Polygon(
            [
                [
                    [self.min_lon, self.min_lat],
                    [self.min_lon, self.max_lat],
                    [self.max_lon, self.max_lat],
                    [self.max_lon, self.min_lat],
                ]
            ]
        )


class SentinelExporter(BaseExporter):
    """Download *cloudless* images from google earth engine,
    leveraging
    https://www.researchgate.net/publication/335850209_Aggregating_cloud-free_Sentinel-2_images_with_Google_Earth_Engine
    """

    image_collection = "COPERNICUS/S2"
    scale = 0.0001
    coordinates = "EPSG:4326"

    def __init__(self, data_dir: Path, country_code: str) -> None:
        self.country_code = country_code
        self.dataset = make_sentinel_dataset_name(country_code)
        super().__init__(data_dir)
        raw_hrsl_dir = data_dir / "raw" / make_hrsl_dataset_name(country_code)
        assert raw_hrsl_dir.exists()

        self.bounding_box = self.get_boundaries(country_code)

    def get_boundaries(self, country_code: str) -> BoundingBox:
        raw_hrsl_path = (
            self.data_dir
            / f"raw/{make_hrsl_dataset_name(country_code)}/hrsl_{country_code}_settlement.tif"
        )

        da = xr.open_rasterio(raw_hrsl_path).isel(band=0)

        return BoundingBox(
            min_lon=da.x.values.min(),
            max_lon=da.x.values.max(),
            min_lat=da.y.values.min(),
            max_lat=da.y.values.max(),
        )

    @staticmethod
    def _monitor(task: ee.batch.Export) -> None:

        while task.status()["state"] in ["READY", "RUNNING"]:
            print(f"Running: {task.status()['state']}")

    @staticmethod
    def _get_bands(bands_type: str) -> Tuple[List[str], int]:
        # the dividor (the int being returned)
        if bands_type == "all":
            return (
                [
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "B7",
                    "B8",
                    "B8A",
                    "B9",
                    "B10",
                    "B11",
                    "B12",
                ],
                1,
            )
        elif bands_type == "TrueRGB":
            return ["B4", "B2", "B3"], int(10000 * 2.5)
        elif bands_type == "RGB":
            # https://code.earthengine.google.com/6051c2f82cdd96b1471f9a26d4210e99
            return ["B12", "B11", "B4"], int(10000 * 2.5)
        else:
            raise AssertionError(f"Unknown bands type {bands_type}")

    def export(self, **kwargs) -> None:
        r"""Export cloud free sentinel data

        :param start_date:
        :param end_date:
        :param monitor:
        :param bands:
        """
        image = cloudfree.get_single_image(
            region=self.bounding_box.to_polygon(),
            start_date=kwargs.get("start_date", date(2018, 12, 1)),
            end_date=kwargs.get("end_date", date(2019, 1, 1)),
        )

        bands_type = kwargs.get("bands", "TrueRGB")
        bands, dividor = self._get_bands(bands_type)

        cloudfree.export(
            image=image.select(bands).divide(dividor),
            region=self.bounding_box.to_polygon(),
            filename=f"{self.country_code}_sentinel_cloud_free_{bands_type}",
            drive_folder=self.dataset,
            monitor=kwargs.get("monitor", False),
        )
