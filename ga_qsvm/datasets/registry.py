from .split import (
    prepare_cancer_data_split,
    prepare_digits_data_split,
    prepare_wine_data_split,
)


DATASET_LOADERS = {
    "wine": prepare_wine_data_split,
    "digits": prepare_digits_data_split,
    "cancer": prepare_cancer_data_split,
}


def get_dataset_loader(name: str):
    try:
        return DATASET_LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc
