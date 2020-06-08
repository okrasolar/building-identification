from pathlib import Path
import xarray as xr
import numpy as np

from .utils import make_sentinel_dataset_name, make_hrsl_dataset_name, load_sentinel

from typing import cast, Tuple, Union


class Engineer:
    def __init__(self, data_dir: Path) -> None:

        self.raw_folder = data_dir / "raw"
        self.processed_folder = data_dir / "processed"

        self.output_folder = data_dir / "features"
        self.output_folder.mkdir(exist_ok=True)

        self.without_buildings = self.output_folder / "without_buildings"
        self.with_buildings = self.output_folder / "with_buildings"

        self.without_buildings.mkdir(exist_ok=True)
        self.with_buildings.mkdir(exist_ok=True)

    def _check_necessary_files_exist(
        self, country_code: str
    ) -> Tuple[Path, Path, bool]:
        hrsl_data = self.processed_folder / make_hrsl_dataset_name(country_code)
        sentinel_data = self.raw_folder / make_sentinel_dataset_name(country_code)

        return sentinel_data, hrsl_data, (hrsl_data.exists() and sentinel_data.exists())

    @staticmethod
    def sentinel_to_numpy(ds: xr.Dataset) -> np.ndarray:
        r"""Return a sentinel ds prepared by the preprocessor as a numpy array
        with dimensions [lat, lon, channels]
        """
        array = ds.transpose("lat", "lon", "band").to_array().values

        return np.squeeze(array, 0)

    @staticmethod
    def hrsl_to_numpy(ds: xr.Dataset) -> np.ndarray:
        r"""Return a hrsl ds prepared by the preprocessor as a numpy array
        with dimensions [lat, lon]
        """
        array = ds.transpose("lat", "lon").to_array().values
        return np.squeeze(array, 0)

    def process_single_filepair(
        self,
        country_code: str,
        hrsl_filepath: Path,
        sentinel_filepath: Path,
        imsize: Union[int, Tuple[int, int]],
        val_ratio: float,
    ) -> None:

        if isinstance(imsize, int):
            imsize = (imsize, imsize)

        lat_imsize, lon_imsize = cast(Tuple, imsize)

        hrsl_ds = xr.open_dataset(hrsl_filepath)
        sentinel_ds = load_sentinel(sentinel_filepath)

        filename = hrsl_filepath.name

        assert hrsl_ds.lat.size == sentinel_ds.lat.size
        assert hrsl_ds.lon.size == sentinel_ds.lon.size

        hrsl_np = self.hrsl_to_numpy(hrsl_ds)
        sentinel_np = self.sentinel_to_numpy(sentinel_ds)

        skipped_files = counter = cur_lat = 0
        max_lat, max_lon = hrsl_np.shape[0], hrsl_np.shape[1]

        while (cur_lat + lat_imsize) <= max_lat:
            cur_lon = 0
            while (cur_lon + lon_imsize) <= max_lon:
                hrsl_slice = hrsl_np[
                    cur_lat : cur_lat + lat_imsize, cur_lon : cur_lon + lon_imsize
                ]
                sentinel_slice = sentinel_np[
                    cur_lat : cur_lat + lat_imsize, cur_lon : cur_lon + lon_imsize
                ]

                if not np.isnan(sentinel_slice).any():

                    self.save_arrays(
                        hrsl_slice,
                        sentinel_slice,
                        country_code,
                        filename,
                        counter,
                        val_ratio,
                    )

                    counter += 1
                else:
                    skipped_files += 1
                cur_lon += lon_imsize
            cur_lat += lat_imsize

        print(f"Saved {counter} files, skipped {skipped_files}")

    def save_arrays(
        self,
        hrsl_slice: np.ndarray,
        sentinel_slice: np.ndarray,
        country_code: str,
        filename: str,
        file_idx: int,
        val_ratio: float,
    ) -> None:

        is_val = np.random.uniform() <= val_ratio

        if is_val:
            data_subset = "validation"
        else:
            data_subset = "training"
        foldername = f"{country_code}_{filename}_{file_idx}"

        contains_buildings = hrsl_slice.max() >= 1

        if contains_buildings:
            pair_folder = self.with_buildings / data_subset / foldername
            pair_folder.mkdir(exist_ok=True, parents=True)
        else:
            pair_folder = self.without_buildings / data_subset / foldername
            pair_folder.mkdir(exist_ok=True, parents=True)

        if contains_buildings:
            # otherwise, this is just an array of 0s - no point
            # in saving it
            np.save(pair_folder / "y.npy", hrsl_slice)
        np.save(pair_folder / "x.npy", sentinel_slice)

        print(f"Saved {foldername}")

    def process_country(
        self,
        country_code: str,
        imsize: Union[int, Tuple[int, int]] = 224,
        val_ratio: float = 0.2,
    ) -> None:

        sentinel_folder, hrsl_folder, files_exist = self._check_necessary_files_exist(
            country_code
        )

        if not files_exist:
            print(f"Missing folders for {country_code}! Skipping")
            return None

        for sentinel_file in sentinel_folder.glob("**/*"):
            if sentinel_file.name.endswith(".tif"):
                hrsl_file = hrsl_folder / sentinel_file.name
                assert hrsl_file.exists()

                print(f"Processing {sentinel_file.name}")
                self.process_single_filepair(
                    country_code,
                    hrsl_file,
                    sentinel_file,
                    imsize=imsize,
                    val_ratio=val_ratio,
                )
            else:
                print(f"Skipping {sentinel_file.name}")
