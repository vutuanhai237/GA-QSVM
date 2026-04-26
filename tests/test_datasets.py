from ga_qsvm.datasets.registry import get_dataset_loader


def test_dataset_loader_registry_contains_supported_runtime_datasets():
    assert get_dataset_loader("wine").__name__ == "prepare_wine_data_split"
    assert get_dataset_loader("digits").__name__ == "prepare_digits_data_split"
    assert get_dataset_loader("cancer").__name__ == "prepare_cancer_data_split"


def test_digits_loader_respects_requested_sizes():
    loader = get_dataset_loader("digits")
    x_train, x_test, y_train, y_test = loader(
        training_size=40,
        test_size=20,
        n_features=5,
        random_state=55,
    )

    assert len(x_train) == 40
    assert len(x_test) == 20
    assert len(y_train) == 40
    assert len(y_test) == 20
    assert x_train.shape[1] == 5
    assert x_test.shape[1] == 5
