import pytest
import xarray as xr
import numpy as np

from src.processors import HRSLProcessor


class TestHRSLProcessor:
    def test_init(self, tmp_path):

        (tmp_path / "raw/hrsl_test_v1").mkdir(parents=True)

        HRSLProcessor(tmp_path, country_code="test")
        assert (tmp_path / "processed/hrsl_test_v1").exists()

    @pytest.mark.parametrize("zeros", [255, 2888, 3979])
    def test_zero_array(self, zeros):

        data = xr.DataArray(np.array([1, zeros, 1, 1, 1, zeros, 1]))

        cleaned_data = HRSLProcessor.zero_array(data)

        assert (cleaned_data.values == np.array([1, 0, 1, 1, 1, 0, 1])).all()
