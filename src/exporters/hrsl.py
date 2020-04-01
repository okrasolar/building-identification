from pathlib import Path
import urllib.request
import zipfile

from .base import BaseExporter
from ..utils import make_hrsl_dataset_name

from typing import Tuple


class HRSLExporter(BaseExporter):
    def __init__(self, data_dir: Path, country_code: str) -> None:
        self.dataset = make_hrsl_dataset_name(country_code)

        super().__init__(data_dir)

        self.url, self.filename = self._make_url(country_code)

    @staticmethod
    def _make_url(country_code: str) -> Tuple[str, str]:

        filename = f"hrsl_{country_code}_v1.zip"

        return f"https://ciesin.columbia.edu/repository/hrsl/{filename}", filename

    def export(self, **kwargs) -> None:

        print(f"Downloading {self.filename}")
        urllib.request.urlretrieve(self.url, self.raw_data_dir / self.filename)

        print(f"Downloaded! Unzipping to {self.raw_data_dir}")
        with zipfile.ZipFile(self.raw_data_dir / self.filename, "r") as zip_file:
            zip_file.extractall(self.raw_data_dir)

        if kwargs.get("remove_zip", False):
            print("Deleting zip file")
            (self.raw_data_dir / self.filename).unlink()
