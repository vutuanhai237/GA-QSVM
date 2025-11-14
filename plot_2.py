#%%
from qiskit.qpy import dump, load
import seaborn as sns
from qiskit.visualization import circuit_drawer
import os

n_features = 10
home = '/home/qsvm/benchmark/GA-QSVM/'
header = 'PQK-digits-N5-Cnot'

# Create output folder for plots
output_folder = os.path.join(home, f'{header}-plots')
os.makedirs(output_folder, exist_ok=True)

# Loop through all folders that begin with header
for folder in os.listdir(home):
    if folder.startswith(header):
        circuit_path = os.path.join(home, folder, 'best_circuit.qpy')
        if os.path.exists(circuit_path):
            print(f"\nLoading circuit from: {folder}")
            with open(circuit_path, 'rb') as f:
                loaded_circuits = load(f)  # qpy.load returns a list of circuits
            
            qc_loaded = loaded_circuits[0]
            
            sns.set_style("whitegrid")
            
            custom_style = {
                'name': 'clifford',  # Giữ style IQP như bạn đang dùng
                'subfontsize': -1,  # Ẩn tham số (theta[2], etc.) bằng cách làm font size = 0
                'fontsize': 12,  # Tùy chọn: Điều chỉnh font chính nếu cần
                'displaycolor': {  # Tùy chỉnh màu cho từng gate
                    'ry': ('#1194FE', '#FFFFFF'),  # RY: Nền đỏ, chữ trắng
                    'rx': ('#F7CD03', '#FFFFFF'),  # RX: Nền xanh dương, chữ đen
                    'rz': ('#00C67B', '#FFFFFF'),
                    'h': ('#00B8B8', '#FFFFFF'),   # H: Nền xanh lá, chữ trắng
                    'cx': ('#E78DAF', '#FFFFFF'),  # CX: Nền xanh lá, chữ đen
                }
            }
            
            # Save plot with folder name
            plot_path = os.path.join(output_folder, f'{folder}.svg')
            qc_loaded.draw( 
                output = 'mpl', 
                style=custom_style, 
                plot_barriers= True, 
                initial_state = False, 
                scale = 1, 
                justify='left',
                reverse_bits=False, 
                filename=plot_path,
            )
            print(f"Circuit plot saved as {plot_path}")
