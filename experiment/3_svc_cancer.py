import sys
# sys.path.append("..")
import wandb
from sklearn.svm import SVC
# from data.cv import prepare_cancer_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

# Test different numbers of qubits (which determines feature dimensions)
max_qubits = 8  
training_size = 100
test_size = 100

# Initialize wandb
wandb.init(project="SVM-PCA", name=f"cancer-SVM-100-100")

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
worst_accuracies = {}
worst_seeds = {}
for num_qubits in range(3, max_qubits + 1):
    # Track worst accuracies for each number of qubits
    
    for random_state in range(1):
        Xw_train, Xw_test, yw_train, yw_test = prepare_cancer_data_split(
            training_size=training_size,
            test_size=test_size,
            n_features=num_qubits,
            random_state=52,
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
        # best_params_list.append(best_params)
        
        # Train SVM with best parameters
        # clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma']).fit(Xw_train, yw_train)
        clf = SVC(kernel='rbf').fit(Xw_train, yw_train)
        # Get accuracy scores
        train_accuracy = clf.score(Xw_train, yw_train)
        test_accuracy = clf.score(Xw_test, yw_test)
        
        # Track worst accuracy for this number of qubits
        if num_qubits not in worst_accuracies or test_accuracy < worst_accuracies[num_qubits]:
            worst_accuracies[num_qubits] = test_accuracy
            worst_seeds[num_qubits] = random_state
        
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)
        
        print(f"Num qubits: {num_qubits}, Features: {num_qubits}, Random state: {random_state}")
        # print(f"Best parameters: C={best_params['C']:.6f}, gamma={best_params['gamma']:.6f}, kernel={best_params['kernel']}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print("-" * 30)
        
        # Calculate average best parameters
        # avg_C = np.mean([params['C'] for params in best_params_list])
        # avg_gamma = np.mean([params['gamma'] for params in best_params_list])
        
        # Log to wandb
        wandb.log({
            "num_qubits": num_qubits,
            "random_state": random_state,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            # "std_train_accuracy": std_train_accuracy,
            # "std_test_accuracy": std_test_accuracy,
            "n_features": num_qubits,
            # "avg_best_C": avg_C,
            # "avg_best_gamma": avg_gamma
        })
        
        # print(f"\nAverages for {num_qubits} qubits:")
        # print(f"Average Best C: {avg_C:.6f}")
        # print(f"Average Best gamma: {avg_gamma:.6f}")
        # print(f"Average Training accuracy: {avg_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
        # print(f"Average Testing accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        # print("=" * 50)

# Print worst accuracies for each number of qubits
print("\n" + "=" * 50)
print("WORST ACCURACIES FOR EACH NUMBER OF QUBITS:")
print("=" * 50)
for qubits in sorted(worst_accuracies.keys()):
    print(f"Qubits: {qubits}, Worst accuracy: {worst_accuracies[qubits]:.4f}, Seed: {worst_seeds[qubits]}")

wandb.finish()