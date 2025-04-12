from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

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
    X, y = shuffle(wine.data, wine.target, random_state=23)
    
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

def prepare_digits_data(training_size, test_size, n_features, machine_id=0, num_machines=3):
    """
    Prepare Digits dataset for binary classification with machine-specific validation sets
    
    Args:
        training_size: Number of samples for training
        test_size: Number of samples for testing
        n_features: Number of features to reduce to using PCA
        machine_id: ID of the current machine (0 to num_machines-1)
        num_machines: Total number of machines for distributed validation
    
    Returns:
        Preprocessed training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Load Digits Dataset
    digits = load_digits()
    X, y = shuffle(digits.data, digits.target, random_state=23)
    
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
    # TODO: Check if MinMaxScaler or StandardScalar or / 255.0 is appropriate for this dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Limit to requested sizes
    if training_size and training_size < len(X_train):
        X_train = X_train[:training_size]
        y_train = y_train[:training_size]
        
    if test_size and test_size < len(X_test):
        X_test = X_test[:test_size]
        y_test = y_test[:test_size]
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def prepare_mnist_data(training_size, test_size, n_features, num_machines=None, machine_id=None, binary=True):
    """
    Prepare MNIST dataset using the official train/test split
    
    Args:
        training_size: Number of samples for training (if None, use all available)
        test_size: Number of samples for testing (if None, use all available)
        n_features: Number of features to reduce to using PCA
        binary: If True, use only digits 0 and 1 for binary classification
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed training and testing datasets
    """
    # Load MNIST Dataset with its official train/test split
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape to vectors
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    
    # Filter for binary classification if requested
    if binary:
        train_mask = (y_train == 0) | (y_train == 1)
        test_mask = (y_test == 0) | (y_test == 1)
        
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]
        
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y_train = 2 * (y_train == 1) - 1  # This converts 0->-1 and 1->1
        y_test = 2 * (y_test == 1) - 1    # This converts 0->-1 and 1->1
    
    x_train, y_train = shuffle(x_train, y_train, random_state=machine_id)
    
    # Normalize data
    X_train = x_train / 255.0
    X_test = x_test / 255.0
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Limit to requested sizes if specified
    if training_size and training_size < len(X_train):
        X_train = X_train[:training_size]
        y_train = y_train[:training_size]
        
    if test_size and test_size < len(X_test):
        X_test = X_test[:test_size]
        y_test = y_test[:test_size]
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test