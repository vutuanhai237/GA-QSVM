import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
# from data.cv import prepare_wine_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


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

# Initialize wandb
wandb.init(project="SVM-PCA", name="wine-SVC-100-78")

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 13
training_size = 100
test_size = 78
num_machines = 10

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
worst_cases = []

for num_qubits in range(3, max_qubits + 1):
    worst_accuracy = float('inf')
    worst_random_state = None
    
    for random_state in range(1):
        # Prepare data with PCA reduction based on num_qubits
        Xw_train, Xw_test, yw_train, yw_test = prepare_wine_data_split(
            training_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            random_state=20
        )
        
        # Train SVM with best parameters
        clf = SVC(kernel='rbf').fit(Xw_train, yw_train)
        
        # Get accuracy scores
        train_accuracy = clf.score(Xw_train, yw_train)
        test_accuracy = clf.score(Xw_test, yw_test)
                
        print(f"Num qubits: {num_qubits}, Random state: {random_state}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print("-" * 30)

        # Update worst accuracy
        if test_accuracy < worst_accuracy:
            worst_accuracy = test_accuracy
            worst_random_state = random_state
        
        # Log to wandb
        wandb.log({
            "num_qubits": num_qubits,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_features": num_qubits,
        })
    
    worst_cases.append({
        "num_qubits": num_qubits,
        "worst_accuracy": worst_accuracy,
        "worst_random_state": worst_random_state
    })

# Output the worst cases for each num_qubits
for case in worst_cases:
    print(f"Num qubits: {case['num_qubits']}, Worst accuracy: {case['worst_accuracy']:.4f}, Random state: {case['worst_random_state']}")

# Send alert notification when run completes
wandb.run.alert(
    title="Experiment Complete",
    text="The wine SVC experiment has finished running"
)




wandb.finish()