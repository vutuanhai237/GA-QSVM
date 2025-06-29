import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def prepare_digits_data_split(train_size, test_size, n_features, binary=False, random_state=23):
    digits = load_digits()
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
        X, y, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
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

# Initialize wandb
wandb.init(project="SVM-PCA", name="digits-SVC-100-50-binary")

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 8  
training_size = 100
test_size = 100

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
# Track worst random states for each number of qubits
worst_random_states = []

for num_qubits in range(3, max_qubits + 1):
    # Prepare data with PCA reduction based on num_qubits
    worst_accuracy = float('inf')
    worst_random_state = None
    
    for random_state in range(1):
        Xw_train, Xw_test, yw_train, yw_test = prepare_digits_data_split(
            train_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            binary=False,
            random_state=55,
        )
        
        # Perform grid search
        # svm = SVC()
        # grid_search = GridSearchCV(
        #     svm, 
        #     param_grid, 
        #     cv=5,
        #     n_jobs=-1,
        #     scoring='accuracy'
        # )
        # grid_search.fit(Xw_train, yw_train)
        
        # Get best parameters
        # best_params = grid_search.best_params_
        
        # Train SVM with best parameters
        # clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma']).fit(Xw_train, yw_train)
        clf = SVC(kernel='rbf').fit(Xw_train, yw_train)
        
        # Get accuracy scores
        train_accuracy = clf.score(Xw_train, yw_train)
        test_accuracy = clf.score(Xw_test, yw_test)
        
        # Track worst accuracy
        if test_accuracy < worst_accuracy:
            worst_accuracy = test_accuracy
            worst_random_state = random_state
        
        print(f"Num qubits: {num_qubits}, Features: {num_qubits}, Random state: {random_state}")
        # print(f"Best parameters: C={best_params['C']:.6f}, gamma={best_params['gamma']:.6f}, kernel={best_params['kernel']}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print("-" * 30)

        # Log to wandb
        wandb.log({
            "num_qubits": num_qubits,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_features": num_qubits,
            # "random_state": random_state,
        })
    
    print(f"WORST ACCURACY for {num_qubits} qubits: {worst_accuracy:.4f} at random_state={worst_random_state}")
    print("=" * 50)
    
    # Add worst random state to the list
    worst_random_states.append({
        'num_qubits': num_qubits,
        'worst_accuracy': worst_accuracy,
        'worst_random_state': worst_random_state
    })

# Output all worst random states at the end
print("\nSUMMARY OF WORST RANDOM STATES:")
print("=" * 60)
for entry in worst_random_states:
    print(f"Qubits: {entry['num_qubits']}, Worst accuracy: {entry['worst_accuracy']:.4f}, Random state: {entry['worst_random_state']}")

wandb.finish()