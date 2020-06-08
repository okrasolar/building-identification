import sys
from pathlib import Path

sys.path.append("..")

from src import Engineer


def engineer():
    engineer = Engineer(data_dir=Path("../data"))
    engineer.process_country("pri")


if __name__ == "__main__":
    engineer()
