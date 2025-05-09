import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
from data.cv import prepare_digits_data
from sklearn.model_selection import GridSearchCV
import numpy as np

# Initialize wandb
wandb.init(project="SVM-PCA", name="test")

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 8  
training_size = 300
test_size = 0
num_machines = 3

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

for num_qubits in range(3, max_qubits + 1):
    # Initialize arrays to store accuracies for averaging
    train_accuracies = []
    test_accuracies = []
    best_params_list = []
    
    # Loop through different machine IDs
    for machine_id in range(num_machines):
        # Prepare data with PCA reduction based on num_qubits
        Xw_train, Xw_test, yw_train, yw_test = prepare_digits_data(
            training_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            machine_id=machine_id,
            num_machines=num_machines
        )
        
        # Perform grid search
        svm = SVC()
        grid_search = GridSearchCV(
            svm, 
            param_grid, 
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(Xw_train, yw_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        
        # Train SVM with best parameters
        clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma']).fit(Xw_train, yw_train)
        
        # Get accuracy scores
        train_accuracy = clf.score(Xw_train, yw_train)
        test_accuracy = clf.score(Xw_test, yw_test)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Num qubits: {num_qubits}, Machine ID: {machine_id}, Features: {num_qubits}")
        print(f"Best parameters: C={best_params['C']:.6f}, gamma={best_params['gamma']:.6f}, kernel={best_params['kernel']}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print("-" * 30)
    
    # Calculate average accuracies and parameters
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    std_train_accuracy = np.std(train_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    
    # Calculate average best parameters
    avg_C = np.mean([params['C'] for params in best_params_list])
    avg_gamma = np.mean([params['gamma'] for params in best_params_list])
    
    # Log to wandb
    wandb.log({
        "num_qubits": num_qubits,
        "avg_train_accuracy": avg_train_accuracy,
        "avg_test_accuracy": avg_test_accuracy,
        "std_train_accuracy": std_train_accuracy,
        "std_test_accuracy": std_test_accuracy,
        "n_features": num_qubits,
        "avg_best_C": avg_C,
        "avg_best_gamma": avg_gamma
    })
    
    print(f"\nAverages for {num_qubits} qubits:")
    print(f"Average Best C: {avg_C:.6f}")
    print(f"Average Best gamma: {avg_gamma:.6f}")
    print(f"Average Training accuracy: {avg_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
    print(f"Average Testing accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
    print("=" * 50)

wandb.finish()