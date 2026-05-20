from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    raw_x_train: np.ndarray
    raw_x_test: np.ndarray
    seed: int
    preprocess: str


def load_dataset(name: str, *, skip_missing: bool = False) -> DatasetBundle | None:
    normalized = name.lower()
    if normalized == "digits":
        dataset = load_digits()
        return DatasetBundle(normalized, dataset.data, dataset.target)
    if normalized == "wine":
        dataset = load_wine()
        return DatasetBundle(normalized, dataset.data, dataset.target)
    if normalized == "cancer":
        dataset = load_breast_cancer()
        return DatasetBundle(normalized, dataset.data, dataset.target)
    if normalized == "fashion":
        try:
            from tensorflow.keras.datasets import fashion_mnist
        except Exception as exc:
            if skip_missing:
                return None
            raise RuntimeError(
                "Fashion MNIST requires TensorFlow. Re-run with --skip-missing to skip it."
            ) from exc
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x = np.concatenate([x_train, x_test]).reshape(-1, 28 * 28).astype(float)
        y = np.concatenate([y_train, y_test])
        return DatasetBundle(normalized, x, y)
    raise ValueError(f"Unsupported dataset: {name}")


def make_holdout_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float | int,
    n_features: int,
    seed: int,
    preprocess: str = "paper",
) -> PreparedSplit:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return prepare_split(
        x_train,
        x_test,
        y_train,
        y_test,
        n_features=n_features,
        seed=seed,
        preprocess=preprocess,
    )


def prepare_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_features: int,
    seed: int,
    preprocess: str = "paper",
) -> PreparedSplit:
    raw_x_train = np.asarray(x_train).copy()
    raw_x_test = np.asarray(x_test).copy()
    if preprocess not in {"legacy", "paper"}:
        raise ValueError(f"Unsupported preprocess mode: {preprocess}")

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(raw_x_train)
    scaled_test = scaler.transform(raw_x_test)
    n_components = min(n_features, scaled_train.shape[1], scaled_train.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    x_train_pca = pca.fit_transform(scaled_train)
    x_test_pca = pca.transform(scaled_test)
    return PreparedSplit(
        x_train=x_train_pca,
        x_test=x_test_pca,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        raw_x_train=raw_x_train,
        raw_x_test=raw_x_test,
        seed=seed,
        preprocess=preprocess,
    )


def dataset_names(values: Iterable[str]) -> list[str]:
    return [value.lower() for value in values]
