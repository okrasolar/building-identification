import sys
from pathlib import Path

sys.path.append("..")

from src.exporters import HRSLExporter, SentinelExporter, GDriveExporter


def export_hrsl(country_code: str = "pri"):

    exporter = HRSLExporter(data_dir=Path("../data"), country_code=country_code)
    exporter.export(remove_zip=True)


def export_sentinel(country_code: str = "phl"):
    exporter = SentinelExporter(data_dir=Path("../data"), country_code=country_code)
    exporter.export()


def export_gdrive(country_code: str = "phl"):
    exporter = GDriveExporter(data_dir=Path("../data"))
    exporter.export(country_code)


if __name__ == "__main__":
    export_hrsl()
    export_sentinel()
    export_gdrive()
