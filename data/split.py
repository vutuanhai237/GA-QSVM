from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np

def prepare_wine_data_split(training_size, test_size, n_features, binary=False, random_state=20):
    wine = load_wine()
    X, y = wine.data, wine.target

    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # Converts 0 -> -1 and 1 -> 1
        print(f"Filtered for binary classification (0 vs 1). Data shape: {X.shape}")
    else:
        print(f"Using multiclass classification (0-3). Data shape: {X.shape}")

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=training_size, test_size=test_size, random_state=20, shuffle=True, stratify=y
    )

    print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    pca = PCA(n_components=n_features) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    print(f"PCA complete. Number of features: {X_train.shape[1]}")
    print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def prepare_digits_data_split(train_size, test_size, n_features, binary=False, random_state=55):
    digits = load_digits()
    X, y = shuffle(digits.data, digits.target, random_state=55)

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
        X, y, train_size=train_size, test_size=test_size, random_state=55, shuffle=True, stratify=y
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

def prepare_cancer_data_split(training_size, test_size, n_features, random_state=52):
    digits = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size,train_size=training_size, random_state=52, stratify=digits.target)
    
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

# I am not going to use this for now. Using this would make the codebase much larger. I will run this after I have all the results and then start comparing.
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