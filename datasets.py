from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

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

def prepare_wine_data(n):
    """
    Prepare Wine dataset for binary classification
    
    Returns:
        Preprocessed training and testing datasets
    """
    # Load Wine Dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Filter for binary classification (only classes 0 and 1)
    X = X[y != 2]
    y = y[y != 2]
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, y_train, y_test