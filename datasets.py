from sklearn.datasets import load_wine, load_digits, load_breast_cancer
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
    scaler = MinMaxScaler()
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

def prepare_digits_data(training_size, test_size, n_features, machine_id=0, num_machines=3, binary=False):
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
    
    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)
        
        X = X[mask]
        y = y[mask]
        
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # This converts 0->-1 and 1->1
    
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
    import tensorflow as tf
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

def prepare_cancer_data(training_size, test_size, n_features, machine_id=0, num_machines=3, binary=False):
    """
    Prepare Breast Cancer dataset for binary classification with machine-specific validation sets
    
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
    digits = load_breast_cancer()
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
    scaler = MinMaxScaler()
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

def prepare_cancer_data_holdout(training_size, test_size, n_features, machine_id=None, num_machines=None, binary=False):
    """
    Prepare Breast Cancer dataset for binary classification with holdout validation
    
    Args:
        training_size: Number of samples for training
        test_size: Number of samples for testing
        n_features: Number of features to reduce to using PCA
    
    Returns:
        Preprocessed training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Load Digits Dataset
    digits = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size,train_size=training_size, random_state=23) # holdout set
    
    # Scale the features 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def prepare_cancer_data_val_eval(training_size, test_size, n_features, machine_id=None, num_machines=None, binary=False):
    # Load Digits Dataset
    digits = load_breast_cancer()
    X, X_eval, y, y_eval = train_test_split(digits.data, digits.target, test_size=test_size, random_state=23)
    
    # Scale the features 
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_eval = scaler.transform(X_eval)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_features)
    X = pca.fit_transform(X)
    X_eval = pca.transform(X_eval)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=training_size, random_state=24)
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_eval)}, Search size: {len(X_val)}")

    return X_train, X_val, X_eval, y_train, y_val, y_eval

def prepare_digits_data_split(train_size, n_features, binary=False, random_state=23):
    """
    Prepare Digits dataset with a standard train/test split and preprocessing.

    Args:
        train_size (float or int): If float, should be between 0.0 and 1.0 and represent the
                                 proportion of the dataset to include in the train split.
                                 If int, represents the absolute number of train samples.
        n_features (int): Number of features to reduce to using PCA.
        binary (bool): If True, filter for digits 0 and 1, convert labels to -1 and 1.
                       If False (default), use all digits 0-9.
        random_state (int): Controls the shuffling applied to the data before splitting and
                           the split itself for reproducibility.

    Returns:
        tuple: Preprocessed training and testing datasets (X_train, X_test, y_train, y_test)
    """
    # Load Digits Dataset
    digits = load_digits()
    
    # Shuffle dataset once initially (optional, as train_test_split can shuffle)
    # Using shuffle here ensures the same shuffling logic as the original if needed downstream,
    # but train_test_split's shuffle=True is generally sufficient.
    X, y = shuffle(digits.data, digits.target, random_state=random_state)

    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # Converts 0 -> -1 and 1 -> 1
        print(f"Filtered for binary classification (0 vs 1). Data shape: {X.shape}")
    else:
        print(f"Using multiclass classification (0-9). Data shape: {X.shape}")

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, shuffle=True # Ensure split is shuffled
    )

    print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    # Reduce dimensionality using PCA (Fit on training data only!)
    # Add random_state to PCA if using randomized solvers like 'arpack' or 'randomized'
    pca = PCA(n_components=n_features, random_state=random_state) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    print(f"PCA complete. Number of features: {X_train.shape[1]}")
    print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test
