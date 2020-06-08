import sys
from pathlib import Path

sys.path.append("..")

from src.processors import HRSLProcessor


def process_hrsl(country_code: str = "pri") -> None:

    processor = HRSLProcessor(data_dir=Path("../data"), country_code=country_code)

    processor.process()


if __name__ == "__main__":
    process_hrsl("pri")
