from abc import ABC, abstractmethod
from pathlib import Path


class BaseExporter(ABC):

    dataset: str

    def __init__(self, data_dir: Path) -> None:

        self.data_dir = data_dir

        self.raw_data_dir = data_dir / f"raw/{self.dataset}"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(self, **kwargs) -> None:
        raise NotImplementedError
