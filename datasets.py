from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
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

def prepare_wine_data(training_size, test_size, n_features, machine_id=0, num_machines=3):
    """
    Prepare Wine dataset for binary classification with machine-specific validation sets
    
    Args:
        training_size: Number of samples for training
        test_size: Number of samples for testing
        n_features: Number of features to reduce to using PCA
        machine_id: ID of the current machine (0 to num_machines-1)
        num_machines: Total number of machines for distributed validation
    
    Returns:
        Preprocessed training and testing datasets
    """
    # Load Wine Dataset
    wine = load_wine()
    X, y = shuffle(wine.data, wine.target, random_state=42)
    
    # Ensure machine_id is valid
    machine_id = machine_id % num_machines
    
    # Create sequential validation splits based on machine_id
    # Similar to how k-fold cross-validation works
    indices = np.arange(len(X))
    n_samples = len(X)
    fold_size = n_samples // num_machines
    
    # Calculate start and end indices for the test fold
    start_idx = machine_id * fold_size
    end_idx = start_idx + fold_size if machine_id < num_machines - 1 else n_samples
    
    # Create train/test indices
    test_indices = indices[start_idx:end_idx]
    train_indices = np.array([i for i in indices if i not in test_indices])
    
    # Split data using the indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Limit to requested sizes
    # X_train = X_train[:training_size, :]
    # y_train = y_train[:training_size]
    # X_test = X_test[:test_size, :]
    # y_test = y_test[:test_size]
    
    return X_train, X_test, y_train, y_test

# TODO: Modify this function based on id machine
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
    scaler = MinMaxScaler()
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