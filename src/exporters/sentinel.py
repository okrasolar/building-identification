from pathlib import Path
from dataclasses import dataclass
from datetime import date
import math
import xarray as xr

import ee

from .base import BaseExporter
from ..utils import make_sentinel_dataset_name, make_hrsl_dataset_name

from typing import Union, List, Tuple

try:
    ee.Initialize()
except Exception:
    print(
        "This code doesn't work unless you have authenticated your earthengine account"
    )

# these are algorithm settings for the cloud filtering algorithm
cloudThresh = 0.2  # Ranges from 0-1.Lower value will mask more pixels out. Generally 0.1-0.3 works well with 0.2 being used most commonly
cloudHeights = [200, 10000, 250]  # Height of clouds to use to project cloud shadows
irSumThresh = 0.3  # Sum of IR bands to include as shadows within TDOM and the shadow shift method (lower number masks out less)
ndviThresh = -0.1
dilatePixels = 2  # Pixels to dilate around clouds
contractPixels = 1  # Pixels to reduce cloud mask and dark shadows by to reduce inclusion of single-pixel comission errors
erodePixels = 1.5
dilationPixels = 3
cloudFreeKeepThresh = 5
cloudMosaicThresh = 50


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
    def _date_to_string(input_date: Union[date, str]) -> str:
        if isinstance(input_date, str):
            return input_date
        else:
            assert isinstance(input_date, date)
            return input_date.strftime("%Y-%m-%d")

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
            return ["B4", "B2", "B3"], 10000
        elif bands_type == "RGB":
            # https://code.earthengine.google.com/6051c2f82cdd96b1471f9a26d4210e99
            return ["B12", "B11", "B4"], 10000
        else:
            raise AssertionError(f"Unknown bands type {bands_type}")

    def export(self, **kwargs) -> ee.batch.Export:
        r"""Export cloud free sentinel data

        :param start_date:
        :param end_date:
        :param monitor:
        :param bands:
        """

        roi = self.bounding_box.to_polygon()

        dates = ee.DateRange(
            self._date_to_string(kwargs.get("start_date", date(2018, 12, 1))),
            self._date_to_string(kwargs.get("end_date", date(2019, 1, 1))),
        )

        startDate = ee.DateRange(dates).start()
        endDate = ee.DateRange(dates).end()
        imgC = (
            ee.ImageCollection(self.image_collection)
            .filterDate(startDate, endDate)
            .filterBounds(roi)
        )

        imgC = (
            imgC.map(lambda x: x.clip(roi))
            .map(lambda x: x.set("ROI", roi))
            .map(computeS2CloudScore)
            .map(calcCloudStats)
            .map(projectShadows)
            .map(computeQualityScore)
            .sort("CLOUDY_PERCENTAGE")
        )

        cloudFree = mergeCollection(imgC)
        cloudFree = cloudFree.reproject(self.coordinates, None, 10)

        bands_type = kwargs.get("bands", "TrueRGB")
        bands, dividor = self._get_bands(bands_type)
        task = ee.batch.Export.image(
            cloudFree.select(bands).divide(dividor),
            f"{self.country_code}_sentinel_cloud_free_{bands_type}",
            {
                "scale": 10,
                "region": roi,
                "maxPixels": 1e13,
                "driveFolder": self.raw_data_dir.name,
            },
        )

        task.start()

        if kwargs.get("monitor", False):
            self._monitor(task)

        return task


def calcCloudStats(img):
    imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
        ee.Geometry(img.get("system:footprint")).coordinates()
    )

    roi = ee.Geometry(img.get("ROI"))

    intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5))
    cloudMask = img.select(["cloudScore"]).gt(cloudThresh).clip(roi).rename("cloudMask")

    cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

    stats = cloudAreaImg.reduceRegion(
        **{"reducer": ee.Reducer.sum(), "geometry": roi, "scale": 10, "maxPixels": 1e12}
    )

    cloudPercent = (
        ee.Number(stats.get("cloudMask")).divide(imgPoly.area()).multiply(100)
    )
    coveragePercent = ee.Number(intersection.area()).divide(roi.area()).multiply(100)
    cloudPercentROI = ee.Number(stats.get("cloudMask")).divide(roi.area()).multiply(100)

    img = img.set("CLOUDY_PERCENTAGE", cloudPercent)
    img = img.set("ROI_COVERAGE_PERCENT", coveragePercent)
    img = img.set("CLOUDY_PERCENTAGE_ROI", cloudPercentROI)

    return img


def rescale(img, exp, thresholds):
    return (
        img.expression(exp, {"img": img})
        .subtract(thresholds[0])
        .divide(thresholds[1] - thresholds[0])
    )


def computeQualityScore(img):
    score = img.select(["cloudScore"]).max(img.select(["shadowScore"]))

    score = score.reproject("EPSG:4326", None, 20).reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    score = score.multiply(-1)

    return img.addBands(score.rename("cloudShadowScore"))


def computeS2CloudScore(img):
    toa = img.select(
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
        ]
    ).divide(10000)

    toa = toa.addBands(img.select(["QA60"]))

    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2']);

    # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1)

    # Clouds are reasonably bright in the blue and cirrus bands.
    score = score.min(rescale(toa, "img.B2", [0.1, 0.5]))
    score = score.min(rescale(toa, "img.B1", [0.1, 0.3]))
    score = score.min(rescale(toa, "img.B1 + img.B10", [0.15, 0.2]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(rescale(toa, "img.B4 + img.B3 + img.B2", [0.2, 0.8]))

    # Clouds are moist
    ndmi = img.normalizedDifference(["B8", "B11"])
    score = score.min(rescale(ndmi, "img", [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, "img", [0.8, 0.6]))

    # Clip the lower end of the score
    score = score.max(ee.Image(0.001))

    # score = score.multiply(dilated)
    score = score.reduceNeighborhood(
        **{"reducer": ee.Reducer.mean(), "kernel": ee.Kernel.square(5)}
    )

    return img.addBands(score.rename("cloudScore"))


def projectShadows(image):
    meanAzimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    meanZenith = image.get("MEAN_SOLAR_ZENITH_ANGLE")

    cloudMask = image.select(["cloudScore"]).gt(cloudThresh)

    # Find dark pixels
    darkPixelsImg = (
        image.select(["B8", "B11", "B12"]).divide(10000).reduce(ee.Reducer.sum())
    )

    ndvi = image.normalizedDifference(["B8", "B4"])
    waterMask = ndvi.lt(ndviThresh)

    darkPixels = darkPixelsImg.lt(irSumThresh)

    # Get the mask of pixels which might be shadows excluding water
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())

    # Find where cloud shadows should be based on solar geometry
    # Convert to radians
    azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

    # Find the shadows
    def getShadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)

        shadowCastedDistance = zenR.tan().multiply(
            cloudHeight
        )  # Distance shadow is cast
        x = (
            azR.sin().multiply(shadowCastedDistance).multiply(-1)
        )  # /X distance of shadow
        y = (
            azR.cos().multiply(shadowCastedDistance).multiply(-1)
        )  # Y distance of shadow
        return image.select(["cloudScore"]).displace(
            ee.Image.constant(x).addBands(ee.Image.constant(y))
        )

    shadows = ee.List(cloudHeights).map(getShadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()

    # Create shadow mask
    shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

    shadowScore = shadowMask.reduceNeighborhood(
        **{"reducer": ee.Reducer.max(), "kernel": ee.Kernel.square(1)}
    )

    image = image.addBands(shadowScore.rename(["shadowScore"]))

    return image


def dilatedErossion(score):
    # Perform opening on the cloud scores
    score = (
        score.reproject("EPSG:4326", None, 20)
        .focal_min(**{"radius": erodePixels, "kernelType": "circle", "iterations": 3})
        .focal_max(
            **{"radius": dilationPixels, "kernelType": "circle", "iterations": 3}
        )
        .reproject("EPSG:4326", None, 20)
    )

    return score


def mergeCollection(imgC):
    # Select the best images, which are below the cloud free threshold, sort them in reverse order
    # (worst on top) for mosaicing
    best = imgC.filterMetadata(
        "CLOUDY_PERCENTAGE", "less_than", cloudFreeKeepThresh
    ).sort("CLOUDY_PERCENTAGE", False)
    filtered = imgC.qualityMosaic("cloudShadowScore")

    # Add the quality mosaic to fill in any missing areas of the ROI which aren't covered by good
    # images
    newC = ee.ImageCollection.fromImages([filtered, best.mosaic()])

    return ee.Image(newC.mosaic())
