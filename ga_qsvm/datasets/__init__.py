from .registry import DATASET_LOADERS, get_dataset_loader
from .split import (
    prepare_cancer_data_split,
    prepare_digits_data_split,
    prepare_wine_data_split,
)

__all__ = [
    "DATASET_LOADERS",
    "get_dataset_loader",
    "prepare_wine_data_split",
    "prepare_digits_data_split",
    "prepare_cancer_data_split",
]
