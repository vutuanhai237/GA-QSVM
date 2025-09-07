# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')   

# Data for the first plot (Breast Cancer)
data_cancer = {
    'Qubit': [4, 5, 6, 7],
    'GA-FQK': [91.5, 93.2, 94.2, 92.6],
    'GA-PQK': [94.3, 95, 95.1, 94.6],
    'FQK': [75.5, 77.0, 76.7, 77.5],
    'PQK': [83.0, 83.4, 83.8, 83.8],  
    'RBF': [94.5, 95.2, 94.7, 95.0]
}
df_cancer = pd.DataFrame(data_cancer).set_index('Qubit')

# Data for the second plot (Digits)
data_digits = {
    'Qubit': [4, 5, 6, 7],
    'GA-FQK': [76.5, 79.1, 83.1, 84.5],
    'GA-PQK': [79, 83.2, 86.7, 88.2],
    'FQK': [54.7, 58.3, 63.5, 67.0],
    'PQK': [56.1, 59, 61.6, 63.4],  
    'RBF': [78.3, 82.6, 84.7, 87.3]
}
df_digits = pd.DataFrame(data_digits).set_index('Qubit')

# Data for the third plot (Wine)
data_wine = {
    'Qubit': [4, 5, 6, 7],
    'GA-FQK': [96.41, 97.05, 97.44, 97.43],
    'GA-PQK': [94.36, 96.67, 97.31, 97.87],
    'FQK': [90.51, 90.51, 90.9, 91.15],
    'PQK': [91.28, 92.56, 92.44, 92.82],
    'RBF': [96.54, 96.41, 97.18, 97.69]
}
df_wine = pd.DataFrame(data_wine).set_index('Qubit')

# Set seaborn style
sns.set_theme(style="white")

# Define the markers for each algorithm
markers = {'GA-FQK': 'o', 'GA-PQK': 'D', 'PQK': '^', 'FQK': 'v', 'RBF': 's'}
dashes = {'GA-FQK': (3, 4, 1, 4), 'GA-PQK': (6,2), 'PQK': (2 , 1), 'FQK': (6 , 2), 'RBF': (2 , 1)}

fig, axes = plt.subplots(1, 3, figsize=(19, 11), sharex=True, sharey=True)

# --- Plot 1: Breast Cancer ---
sns.lineplot(data=df_cancer, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[0], legend=False)
axes[0].set_title('Breast Cancer Dataset', fontsize=23, y=0.96)
axes[0].set_ylabel('Accuracy (%)', fontsize=23)
axes[0].set_xlabel('Number of Qubits', fontsize=23)
# axes[0].set_xmargin(0)



# --- Plot 2: Digits ---
sns.lineplot(data=df_digits, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[1], legend=False)
axes[1].set_title('Digits Dataset', fontsize=23, y=0.96)
axes[1].set_xlabel('Number of Qubits', fontsize=23)
axes[1].set_ylabel('') # Remove redundant y-axis label
# axes[1].set_xmargin(0)
# --- Plot 3: Wine ---
# We plot with a legend here so we can grab the handles and labels from it
sns.lineplot(data=df_wine, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[2])
axes[2].set_title('Wine Dataset', fontsize=23, y=0.96)
axes[2].set_ylabel('') # Remove redundant y-axis label
axes[2].set_xlabel('Number of Qubits', fontsize=23)
# axes[2].set_xmargin(0)

# Set integer ticks for the shared x-axis
for ax in axes:
    ax.set_xticks(ticks=df_cancer.index)
    ax.set_xticklabels(labels=df_cancer.index, fontsize=23)

y_ticks = np.arange(50, 101, 10)
axes[0].set_yticks(y_ticks)
axes[0].tick_params(axis='y', labelsize=23) # Change 12 to your desired font size

for ax in axes:
    ax.grid(axis='both', linestyle='--', alpha=0.7)
# plt.grid(axis='both', linestyle='--')

# Create a single shared legend from the last plot (which has the legend data)
handles, labels = axes[2].get_legend_handles_labels()
axes[2].get_legend().remove() # Remove the individual legend from the third plot
axes[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)
axes[1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.25, 0.67), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)
axes[2].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)

# Adjust layout to prevent titles and legend from overlapping
plt.tight_layout()
plt.subplots_adjust(bottom=0.15) # Make space for the bottom legend

plt.savefig("generalize.pdf")
plt.show()

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Extraction ---

# Data for the Wine Dataset
data_wine = {
    'Qubit': list(range(3, 8)),
    'GA-FQK': [95, 95, 97, 99, 97],
    'GA-PQK': [94.872, 96.154, 96.154, 97.436, 98.718],
    'FQK': [90, 91, 91, 92, 92],  # ZFeatureMap data
    'PQK': [91, 90, 92, 91, 91],
    'RBF': [91, 91, 94, 94, 92],
}
df_wine = pd.DataFrame(data_wine).set_index('Qubit')

# Data for the Breast Cancer Dataset
data_cancer = {
    'Qubit': list(range(3, 8)),
    'GA-FQK': [91, 94, 95, 96, 97],
    'GA-PQK': [93, 92, 92, 95, 97],
    'FQK': [66, 71, 69, 72, 74],  # ZFeatureMap data
    'PQK': [74, 73, 73, 73, 73],
    'RBF': [87, 91, 90, 90, 90],
}
df_cancer = pd.DataFrame(data_cancer).set_index('Qubit')

# Data for the Digits Dataset
data_digits = {
    'Qubit': list(range(3, 8)),
    'GA-FQK': [64, 75, 77, 81, 82],
    'GA-PQK': [65, 77, 78, 84, 86],
    'FQK': [51, 53, 61, 63, 58],  # ZFeatureMap data
    'PQK': [51, 56, 59, 59, 58],
    'RBF': [64, 72, 74, 80, 77],
}
df_digits = pd.DataFrame(data_digits).set_index('Qubit')


# Set seaborn style
sns.set_theme(style="white")

# markers = {'GA-FQK': 'o', 'GA-PQK': 'D', 'Z-Feature': '^', 'RBF': 's'}
# dashes = {'GA-FQK': (3, 4, 1, 4), 'GA-PQK': (6,2), 'Z-Feature': (2 , 1), 'RBF': (2 , 1)}

# Define the markers for each algorithm
markers = {'GA-FQK': 'o', 'GA-PQK': 'D', 'FQK': '^', 'PQK': 'v', 'RBF': 's'}
dashes = {'GA-FQK': (3, 4, 1, 4), 'GA-PQK': (6,2), 'FQK': (1, 1), 'PQK': (6, 2), 'RBF': (2 , 1)}

fig, axes = plt.subplots(1, 3, figsize=(19, 12), sharex=True, sharey=True)

# --- Plot 1: Breast Cancer ---
sns.lineplot(data=df_cancer, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[0], legend=False)
axes[0].set_title('Breast Cancer Dataset', fontsize=23, y=0.96)
axes[0].set_ylabel('Accuracy (%)', fontsize=23)
axes[0].set_xlabel('Number of Qubits', fontsize=23)

# --- Plot 2: Digits ---
sns.lineplot(data=df_digits, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[1], legend=False)
axes[1].set_title('Digits Dataset', fontsize=23, y=0.96)
axes[1].set_xlabel('Number of Qubits', fontsize=23)
axes[1].set_ylabel('') # Remove redundant y-axis label

# --- Plot 3: Wine ---
# We plot with a legend here so we can grab the handles and labels from it
sns.lineplot(data=df_wine, markers=markers, linewidth=2.5, markersize=24, dashes=dashes, ax=axes[2])
axes[2].set_title('Wine Dataset', fontsize=23, y=0.96)
axes[2].set_ylabel('') # Remove redundant y-axis label
axes[2].set_xlabel('Number of Qubits', fontsize=23)

# Set integer ticks for the shared x-axis
for ax in axes:
    ax.set_xticks(ticks=df_wine.index)
    ax.set_xticklabels(labels=df_wine.index, fontsize=23)

y_ticks = np.arange(30, 101, 10)
axes[0].set_yticks(y_ticks)
axes[0].tick_params(axis='y', labelsize=23)

for ax in axes:
    ax.grid(axis='both', linestyle='--', alpha=0.7)

# Create a single shared legend from the last plot (which has the legend data)
handles, labels = axes[2].get_legend_handles_labels()
axes[2].get_legend().remove() # Remove the individual legend from the third plot
axes[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)
axes[1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.29, 0.7), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)
axes[2].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1, fontsize=23, title="", numpoints=1, title_fontsize=15)

# Adjust layout to prevent titles and legend from overlapping
plt.tight_layout()
plt.subplots_adjust(bottom=0.15) # Make space for the bottom legend

plt.savefig("unfair.pdf")
plt.show()

# %%

