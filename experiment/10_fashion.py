from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def prepare_fashion_mnist_data_split(training_size, test_size, n_features, binary=False, random_state=55):
    # Load Fashion MNIST dataset
    (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
    
    # Combine training and test sets
    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)
    
    # Reshape from 28x28 images to flat vectors (784 features)
    X = X.reshape(X.shape[0], -1)
    
    # Convert to float and normalize to 0-1 range
    X = X.astype('float32') / 255.0
    
    # Shuffle the data
    X, y = shuffle(X, y, random_state=random_state)

    # Filter for binary classification if requested
    if binary:
        mask = (y == 0) | (y == 1)  # T-shirt/top vs Trouser
        X = X[mask]
        y = y[mask]
        # Convert to binary labels (-1 for class 0, 1 for class 1)
        y = 2 * (y == 1) - 1  # Converts 0 -> -1 and 1 -> 1
        # print(f"Filtered for binary classification (T-shirt/top vs Trouser). Data shape: {X.shape}")
    # else:
        # print(f"Using multiclass classification (10 fashion categories). Data shape: {X.shape}")

    # Split data into training and testing sets BEFORE scaling/PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=training_size, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    # print(f"Split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale the features (Fit on training data only!)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Transform test data using training scaler

    # Reduce dimensionality using PCA (Fit on training data only!)
    # Add random_state to PCA if using randomized solvers like 'arpack' or 'randomized'
    pca = PCA(n_components=n_features, random_state=random_state) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) # Transform test data using training PCA

    # print(f"PCA complete. Number of features: {X_train.shape[1]}")
    # print(f"Final Training size: {len(X_train)}, Final Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def benchmark_rbf_svm_seeds(max_seeds=1000):
    """
    Benchmark RBF SVM performance across different random seeds to find the worst seed.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        max_seeds: Maximum number of seeds to test (default: 1000)
    
    Returns:
        worst_seed: The seed with the worst performance
        worst_score: The worst accuracy score
        all_results: Dictionary with seed -> score mapping
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import time
    
    print(f"Starting RBF SVM benchmark across {max_seeds} seeds...")
    start_time = time.time()
    
    all_results = {}
    worst_score = float('inf')
    worst_seed = None
    
    for seed in range(max_seeds):
        # Set random state for reproducibility
        X_train, X_test, y_train, y_test = prepare_fashion_mnist_data_split(
            training_size=100, 
            test_size=100, 
            n_features=10,  # Reduced features for faster computation
            random_state=seed
        )
    
        svm = SVC(kernel='rbf')
        
        # Train the model
        svm.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = svm.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        all_results[seed] = score
        
        # Track worst performance
        if score < worst_score:
            worst_score = score
            worst_seed = seed
        
        # Progress update every 100 seeds
        if (seed + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {seed + 1}/{max_seeds} seeds. "
                  f"Current worst: seed {worst_seed} (score: {worst_score:.4f}). "
                  f"Elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nBenchmark complete in {total_time:.1f} seconds")
    print(f"Worst performing seed: {worst_seed} with accuracy: {worst_score:.4f}")
    
    # Print some statistics
    scores = list(all_results.values())
    print(f"Mean accuracy: {np.mean(scores):.4f}")
    print(f"Std accuracy: {np.std(scores):.4f}")
    print(f"Min accuracy: {np.min(scores):.4f}")
    print(f"Max accuracy: {np.max(scores):.4f}")
    
    return worst_seed, worst_score, all_results


# Example usage:
if __name__ == "__main__":
    # Run the benchmark
    worst_seed, worst_score, all_results = benchmark_rbf_svm_seeds()

