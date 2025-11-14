#%%
from qiskit.qpy import dump, load
n_features = 10
import os

home = '/home/qsvm/GA-QSVM/'
header = 'PQK-digits-N7'

# Loop through all folders that begin with header
for folder in os.listdir(home):
    if folder.startswith(header):
        circuit_path = os.path.join(home, folder, 'best_circuit.qpy')
        if os.path.exists(circuit_path):
            print(f"\nLoading circuit from: {folder}")
            with open(circuit_path, 'rb') as f:
                loaded_circuits = load(f)  # qpy.load returns a list of circuits
            
            qc_loaded = loaded_circuits[0]
            
            # Create header folder if it doesn't exist
            header_folder = os.path.join(home, header)
            os.makedirs(header_folder, exist_ok=True)
            
            # Save circuit as text with folder name
            text_path = os.path.join(header_folder, f'{folder}.txt')
            with open(text_path, 'w') as f:
                f.write(str(qc_loaded))
            print(f"Circuit saved as {text_path}")