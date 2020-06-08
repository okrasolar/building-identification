import sys
from pathlib import Path

sys.path.append("..")

from src.exporters import HRSLExporter, SentinelExporter


def export_hrsl(country_code: str = "pri"):

    exporter = HRSLExporter(data_dir=Path("../data"), country_code=country_code)
    exporter.export(remove_zip=True)


def export_sentinel(country_code: str = "pri"):
    exporter = SentinelExporter(data_dir=Path("../data"), country_code=country_code)
    exporter.export()


if __name__ == "__main__":
    export_hrsl()
    export_sentinel()
