from ga_qsvm.datasets.registry import get_dataset_loader


def test_dataset_loader_registry_contains_supported_runtime_datasets():
    assert get_dataset_loader("wine").__name__ == "prepare_wine_data_split"
    assert get_dataset_loader("digits").__name__ == "prepare_digits_data_split"
    assert get_dataset_loader("cancer").__name__ == "prepare_cancer_data_split"


def test_digits_loader_keeps_baseline_fixed_sizes():
    loader = get_dataset_loader("digits")
    x_train, x_test, y_train, y_test = loader(
        training_size=40,
        test_size=20,
        n_features=5,
        random_state=55,
    )

    assert len(x_train) == 100
    assert len(x_test) == 100
    assert len(y_train) == 100
    assert len(y_test) == 100
    assert x_train.shape[1] == 5
    assert x_test.shape[1] == 5


def test_wine_and_cancer_loaders_keep_baseline_fixed_sizes():
    wine = get_dataset_loader("wine")
    cancer = get_dataset_loader("cancer")

    wine_train, wine_test, _, _ = wine(training_size=40, test_size=20, n_features=3)
    cancer_train, cancer_test, _, _ = cancer(training_size=40, test_size=20, n_features=3)

    assert len(wine_train) == 100
    assert len(wine_test) == 78
    assert len(cancer_train) == 100
    assert len(cancer_test) == 100
