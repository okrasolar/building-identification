import torch

from src.models.data import ClassifierDataset


class TestClassifier:
    def test_y_correctly_created(self, tmp_path):

        num_buildings = 3
        num_non_buildings = 5

        for i in range(num_buildings):
            i_folder = tmp_path / "features" / "with_buildings" / str(i)
            i_folder.mkdir(parents=True)
            (i_folder / "x.npy").touch()

        for i in range(num_non_buildings):
            i_folder = tmp_path / "features" / "without_buildings" / str(i)
            i_folder.mkdir(parents=True)
            (i_folder / "x.npy").touch()

        dataset = ClassifierDataset(tmp_path)

        assert (
            dataset.y == torch.Tensor([1] * num_buildings + [0] * num_non_buildings)
        ).all()
