from pathlib import Path
import random
from matplotlib import pyplot as plt
import numpy as np

from typing import Tuple


COUNTRY_CODES = ["pri", "khm"]


def make_hrsl_dataset_name(country_code: str) -> str:
    return f"hrsl_{country_code}_v1"


def make_sentinel_dataset_name(country_code: str) -> str:
    return f"sentinel_{country_code}"


def plot_subset(
    features_dir: Path,
    num_examples: int = 3,
    figsize: Tuple[int, int] = (15, 5),
    plot_orgs: bool = True,
) -> None:
    r"""Plots a subset of features, with masks.
    Useful for data validation.

    :param features_dir: Path to the features dir
    :param num_examples: The number of examples to plot
    """
    all_files = list(features_dir.glob("*"))

    selected_files = random.choices(all_files, k=num_examples)
    print(selected_files)

    fig, ax = plt.subplots(nrows=1, ncols=num_examples, figsize=figsize)

    for idx, folder in enumerate(selected_files):
        image = np.load(folder / "x.npy")
        mask = np.load(folder / "y.npy")

        ax[idx].imshow(image)
        ax[idx].imshow(mask, alpha=0.3)

    if plot_orgs:
        fig2, ax2 = plt.subplots(nrows=1, ncols=num_examples, figsize=figsize)
        for idx, folder in enumerate(selected_files):
            image = np.load(folder / "x.npy")

            ax2[idx].imshow(image)

    plt.show()
