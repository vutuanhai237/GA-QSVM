from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def generate_data(n_samples, n_features, centers, random_state):
    """
    Generate synthetic data for binary classification using make_blobs
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features for each sample
        centers: Number of centers/clusters
        random_state: Random seed for reproducibility
    
    Returns:
        Preprocessed training and testing datasets
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    y = 2 * y - 1  # Convert labels to {-1, +1} for QSVC compatibility
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def prepare_wine_data(training_size, test_size, n_features):
    """
    Prepare Wine dataset for binary classification
    
    Returns:
        Preprocessed training and testing datasets
    """
    # Load Wine Dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = X_train[:training_size, :]
    y_train = y_train[:training_size]
    X_test = X_test[:test_size, :]
    y_test = y_test[:test_size]
    
    return X_train, X_test, y_train, y_test

def prepare_digits_data(training_size, test_size, n_features):
    """
    Prepare Digits dataset for binary classification
    
    Args:
        n_features: Number of features to reduce to using PCA
    
    Returns:
        Preprocessed training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Load Digits Dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Check if sets of labels is not equal 10
    if len(set(y_train)) != 10:
        return None, None, None, None

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    data = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

    X_train = X_train[:training_size, :]
    y_train = y_train[:training_size]
    X_test = X_test[:test_size, :]
    y_test = y_test[:test_size]
    
    return X_train, X_test, y_train, y_test