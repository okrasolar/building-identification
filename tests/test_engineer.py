import xarray as xr
import numpy as np

from src import Engineer
from src.utils import make_sentinel_dataset_name, make_hrsl_dataset_name

import pytest


class TestEngineer:
    @pytest.mark.parametrize("failure_case", [True, False])
    def test_necessary_files(self, tmp_path, failure_case):

        country = "khm"

        processed_folder = tmp_path / "processed"

        hrsl_folder = processed_folder / make_hrsl_dataset_name(country)
        sentinel_folder = processed_folder / make_sentinel_dataset_name(country)

        if not failure_case:
            hrsl_folder.mkdir(parents=True)
            sentinel_folder.mkdir(parents=True)

        engineer = Engineer(tmp_path)
        (
            returned_sentinel_file,
            returned_hrsl_file,
            files_exist,
        ) = engineer._check_necessary_files_exist(country)

        assert returned_hrsl_file == hrsl_folder
        assert returned_sentinel_file == sentinel_folder
        if failure_case:
            assert files_exist is False
        else:
            assert files_exist is True

    def test_sentinel_to_numpy(self):

        lon_len = 10
        lat_len = 15
        longitudes = np.linspace(0, 10, lon_len)
        latitudes = np.linspace(0, 10, lat_len)

        dims = ["lat", "lon", "band"]
        coords = {"lat": latitudes, "lon": longitudes, "band": [0, 1, 2]}

        var = np.random.randint(100, size=(lat_len, lon_len, 3))

        ds = xr.Dataset({"test_var": (dims, var)}, coords=coords)

        np_ds = Engineer.sentinel_to_numpy(ds)
        assert np_ds.shape == (lat_len, lon_len, 3)

    def test_hrsl_to_numpy(self):

        lon_len = 10
        lat_len = 15
        longitudes = np.linspace(0, 10, lon_len)
        latitudes = np.linspace(0, 10, lat_len)

        dims = ["lat", "lon"]
        coords = {"lat": latitudes, "lon": longitudes}

        var = np.random.randint(100, size=(lat_len, lon_len))

        ds = xr.Dataset({"test_var": (dims, var)}, coords=coords)

        np_ds = Engineer.hrsl_to_numpy(ds)
        assert np_ds.shape == (lat_len, lon_len)

    @pytest.mark.parametrize("with_buildings", [True, False])
    def test_process_single_filepair(self, tmp_path, with_buildings):

        country = "khm"

        lon_len = 4
        lat_len = 2
        longitudes = np.linspace(0, 10, lon_len)
        latitudes = np.linspace(0, 10, lat_len)

        sentinel_var_one_band = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]])

        sentinel_var = np.stack(
            [sentinel_var_one_band, sentinel_var_one_band, sentinel_var_one_band], -1
        )

        if with_buildings:
            mask = np.ones_like(sentinel_var_one_band)
        else:
            mask = np.zeros_like(sentinel_var_one_band)

        hrsl_ds = xr.Dataset(
            {"hrsl": (["lat", "lon"], mask)},
            coords={"lat": latitudes, "lon": longitudes},
        )
        sentinel_ds = xr.Dataset(
            {"sentinel": (["lat", "lon", "band"], sentinel_var)},
            coords={"lat": latitudes, "lon": longitudes, "band": [0, 1, 2]},
        )

        processed_folder = tmp_path / "processed"

        hrsl_folder = processed_folder / make_hrsl_dataset_name(country)
        sentinel_folder = processed_folder / make_sentinel_dataset_name(country)

        hrsl_folder.mkdir(parents=True)
        sentinel_folder.mkdir(parents=True)

        hrsl_ds.to_netcdf(hrsl_folder / "data.nc")
        sentinel_ds.to_netcdf(sentinel_folder / "data.nc")

        engineer = Engineer(tmp_path)

        imsize = 2
        engineer.process_single_filepair(
            country_code=country,
            hrsl_filepath=hrsl_folder / "data.nc",
            sentinel_filepath=sentinel_folder / "data.nc",
            imsize=imsize,
        )

        if with_buildings:
            output_folder = tmp_path / "features/with_buildings"
        else:
            output_folder = tmp_path / "features/without_buildings"

        output_folders = list(output_folder.glob("*"))
        assert len(output_folders) == 2

        for output_folder in output_folders:

            x = np.load(output_folder / "x.npy")

            assert x.shape == (imsize, imsize, 3)

            if output_folder.name.endswith("0"):
                assert (x == np.ones_like(x)).all()
            elif output_folder.name.endswith("1"):
                assert (x == np.ones_like(x) * 2).all()

            if with_buildings:
                y = np.load(output_folder / "y.npy")

                assert (y == np.ones_like(y)).all()
            else:
                assert not (output_folder / "y.npy").exists()
