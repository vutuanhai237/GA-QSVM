from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Define parameter grid
param_grid = {
    'C': np.logspace(-3, 2, 6),
    'gamma': np.logspace(-3, 2, 6),
    'kernel': ['rbf']
}

# Create SVC classifier
base_svc = SVC(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(base_svc, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Print best parameters
print("Best parameters found:")
print(grid_search.best_params_)
print("\nBest score:", grid_search.best_score_)

# Create SVC with best parameters
best_svc = SVC(**grid_search.best_params_, random_state=42)

# Perform 10-fold cross validation with best parameters
scores = cross_val_score(best_svc, X, y, cv=10)

# Print results
print("\nCross-validation scores with best parameters:", scores)
print("Average accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 