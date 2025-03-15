from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define parameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': np.logspace(-4, 1, 6), 
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# PCA
pca = PCA(n_components=3)
X = pca.fit_transform(X)

# Create SVC classifier
base_svc = SVC(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(base_svc, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Print best parameters
print("Best parameters found:")
print(grid_search.best_params_)
print("\nBest score:", grid_search.best_score_)

# Create SVC with best parameters
best_svc = SVC(**grid_search.best_params_, random_state=42)

# Perform cross validation with different random states
print("\nCross-validation scores for different random states:")
for n in range(5):
    strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=n)
    scores = cross_val_score(best_svc, X, y, cv=strat_k_fold)
    print(f"\nRandom state {n}:")
    print("Scores:", scores)
    print("Average accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 