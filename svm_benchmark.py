import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
import os
import wandb
from datetime import datetime

# Import from existing files
from datasets import prepare_wine_data, prepare_digits_data

def run_svm_benchmark(dataset_func, qubit_range, training_size=100, test_size=50, 
                     id=0, num_machines=3, param_grid=None, cv=5, dataset_name="unknown"):
    results = []
    
    if param_grid is None:
        param_grid = {
            'C': np.logspace(-3, 2, 6),
            'gamma': np.logspace(-3, 2, 6),
            'kernel': ['rbf']
        }
    
    for num_qubits in qubit_range:
        run = wandb.init(
            project=f"SVM-Benchmark-{dataset_name}",
            name=f"qubits-{num_qubits}",
            config={"num_qubits": num_qubits, "dataset": dataset_name},
            reinit=True
        )
        
        # Load dataset with specified compression
        X_train, X_test, y_train, y_test = dataset_func(
            training_size, test_size, num_qubits, id, num_machines
        )
        
        if X_train is None or len(np.unique(y_train)) != len(np.unique(y_test)):
            wandb.finish()
            continue
            
        # Record dimensions
        wandb.log({"original_dimensions": X_train.shape[1], "compressed_dimensions": num_qubits})
        
        # Find optimal hyperparameters
        start_time = time.time()
        grid_search = GridSearchCV(
            SVC(), 
            param_grid=param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_train, y_train)
        clf = SVC(**grid_search.best_params_)
        clf.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, clf.predict(X_train)),
            'test_accuracy': accuracy_score(y_test, clf.predict(X_test)),
            'runtime': time.time() - start_time
        }
        
        # Log to wandb
        wandb.log({
            **metrics,
            'cv_score': grid_search.best_score_
        })
        
        results.append({
            'num_qubits': num_qubits,
            'compressed_dims': num_qubits,
            'original_dims': X_train.shape[1],
            **metrics,
            'best_params': str(grid_search.best_params_)
        })
        
        wandb.finish()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    wandb.login()
    
    config = {
        'training_size': 100,
        'test_size': 50,
        'id': 0,
        'num_machines': 3,
        'qubit_range': range(3, 8)
    }
    
    datasets = {
        'wine': (prepare_wine_data, {
            'C': np.logspace(-3, 2, 6),
            'gamma': np.logspace(-3, 2, 6),
            'kernel': ['rbf']
        })
    }
    
    os.makedirs('benchmark_results', exist_ok=True)
    
    for dataset_name, (dataset_func, param_grid) in datasets.items():
        results_df = run_svm_benchmark(
            dataset_func=dataset_func,
            qubit_range=config['qubit_range'],
            training_size=config['training_size'],
            test_size=config['test_size'],
            id=config['id'],
            num_machines=config['num_machines'],
            param_grid=param_grid,
            dataset_name=dataset_name
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'benchmark_results/svm_benchmark_{dataset_name}_{timestamp}.csv', index=False)
