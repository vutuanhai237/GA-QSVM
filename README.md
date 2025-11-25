# Source code for paper entitled "Flexible Genetic Algorithm for Quantum Support Vector Machines"

This repository contains an implementation of a Genetic Algorithm-based Quantum Support Vector Machine (GA-QSVM) for classification tasks. The project combines quantum computing with machine learning techniques to optimize quantum circuits for classification.

Paper: https://arxiv.org/pdf/2511.19160

Bibtex for citation:

```
@misc{duc2025flexiblegeneticalgorithmquantum,
      title={Flexible Genetic Algorithm for Quantum Support Vector Machines}, 
      author={Nguyen Minh Duc and Vu Tuan Hai and Le Bin Ho and Tran Nguyen Lan},
      year={2025},
      eprint={2511.19160},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2511.19160}, 
}
```


## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Main Script](#running-the-main-script)
  - [Command-line Arguments](#command-line-arguments)
  - [Using the Training Scripts](#using-the-training-scripts)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

Key features:
- Quantum circuit optimization using genetic algorithms
- Integration with Qiskit for quantum computing
- Support for multiple classification datasets
- Hyperparameter tuning capabilities
- Distributed training support
- Experiment tracking with Weights & Biases (wandb)

## Requirements

The project requires the following dependencies:
```
numpy==1.26.4
scikit-learn==1.6.0
qiskit==1.3.1
qiskit_machine_learning==0.8.2
wandb==0.19.8
tensorflow==2.18.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vutuanhai237/GA-QSVM.git
cd GA-QSVM
```

2. Create and activate a virtual environment (recommended):
```bash
# Using venv
python -m venv ga-qsvm
source ga-qsvm/bin/activate  # On Windows: ga-qsvm\Scripts\activate

# Using conda
conda create --name ga-qsvm python=3.11
conda activate ga-qsvm
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

The main script (`main.py`) supports various command-line arguments for configuring the experiments:

```bash
python main.py --depth 4 --num-circuit 8 --qubits 3 4 5 --num-machines 3 --id 0 --training-size 300 --test-size 0 --data digits
```

### Command-line Arguments

- `--depth`: Circuit depth(s) to try (default: [4, 5, 6])
- `--num-circuit`: Number of circuits to try in parallel (default: range(4, 33, 4))
- `--num-generation`: Number of generations for genetic algorithm (default: [100])
- `--prob-mutate`: Mutation probabilities to try (default: logarithmic space from 0.01 to 0.1)
- `--qubits`: Number of qubits to try (default: [3, 4, 5, 6, 7, 8])
- `--training-size`: Size of training dataset (default: 100)
- `--test-size`: Size of test dataset (default: 50)
- `--num-machines`: Number of machines for cross-validation (default: 3)
- `--id`: ID for the machine (default: 0)
- `--start-index`: Index to start from in base combinations, ie. when the running fail, use this to continue the benchmarking (default: 0)
- `--data`: Dataset to use ('wine', 'digits', or 'cancer') (default: 'wine')

### Using the Training Scripts

The repository includes training scripts (`train0.sh`, `train1.sh`, `train2.sh`) for running experiments on a HPC cluster. These scripts can be submitted to the cluster using the `qsub` command.

```bash
qsub train0.sh
qsub train1.sh
qsub train2.sh
```

## Datasets

The project supports the following datasets:
- Wine dataset
- Digits dataset
- MNIST dataset
- Breast Cancer dataset

Each dataset can be prepared with different feature dimensions using PCA.

## Project Structure

- `main.py`: Main script for running GA-QSVM experiments
- `datasets.py`: Functions for preparing different datasets
- `utils.py`: Utility functions
- `requirements.txt`: Project dependencies
- `train*.sh`: Training scripts for cluster environments
- `experiment/` : Scripts for experiments
- `notebook/`: Jupyter notebooks for experiments
- `qoop/`: Quantum Object Optimizer package

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project uses the QOOP (Quantum Object Optimizer) package developed by Vu Tuan Hai, Nguyen Tan Viet, and Le Bin Ho.
