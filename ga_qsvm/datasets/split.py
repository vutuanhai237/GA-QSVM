from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _split_scale_project(X, y, training_size, test_size, n_features, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=training_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    pca = PCA(n_components=n_features, random_state=random_state)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test, y_train, y_test


def prepare_wine_data_split(training_size, test_size, n_features, random_state=20):
    dataset = load_wine()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )


def prepare_digits_data_split(training_size, test_size, n_features, random_state=55):
    dataset = load_digits()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )


def prepare_cancer_data_split(training_size, test_size, n_features, random_state=52):
    dataset = load_breast_cancer()
    return _split_scale_project(
        dataset.data,
        dataset.target,
        training_size,
        test_size,
        n_features,
        random_state,
    )
