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
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Command-line Arguments](#command-line-arguments)
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
scipy
matplotlib
tqdm
qiskit==1.3.1
qiskit-machine-learning==0.8.2
wandb==0.19.8
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vutuanhai237/GA-QSVM.git
cd GA-QSVM
```

2. Sync the project environment with `uv`:
```bash
uv sync --dev
```

## Usage

### Training

Run training from the package CLI:

```bash
uv run python -m ga_qsvm.cli.train --depth 4 --num-circuit 8 --qubits 3 4 5 --num-machines 3 --id 0 --training-size 300 --test-size 50 --data digits
```

### Evaluation

Run evaluation from the package CLI:

```bash
uv run python -m ga_qsvm.cli.eval --rx 1 --ry 2 --rz 3 --num-qubits 4 --prob-mutate 0.1 --data wine
```

### Command-line Arguments

Training (`ga_qsvm.cli.train`):
- `--depth`: Circuit depth(s) to try (default: [4, 5, 6])
- `--num-circuit`: Number of circuits to try in parallel (default: range(4, 33, 4))
- `--num-generation`: Number of generations for genetic algorithm (default: [100])
- `--prob-mutate`: Mutation probabilities to try (default: [0.01, 0.1])
- `--qubits`: Number of qubits to try (default: [3, 4, 5, 6, 7, 8])
- `--training-size`: Size of training dataset (default: 100)
- `--test-size`: Size of test dataset (default: 50)
- `--num-machines`: Number of machines for cross-validation (default: 3)
- `--id`: Machine identifier (default: 0)
- `--start-index`: Index to start from in base combinations, ie. when the running fail, use this to continue the benchmarking (default: 0)
- `--data`: Dataset to use ('wine', 'digits', or 'cancer') (default: 'wine')

Evaluation (`ga_qsvm.cli.eval`):
- `--rx`: Number of RX rotations
- `--ry`: Number of RY rotations
- `--rz`: Number of RZ rotations
- `--num-qubits`: Number of qubits
- `--prob-mutate`: Mutation probability
- `--data`: Dataset to use ('digits', 'wine', or 'cancer')

## Datasets

The project supports the following datasets:
- Wine dataset
- Digits dataset
- Breast Cancer dataset

## Project Structure

- `ga_qsvm/cli/`: Package CLIs for training and evaluation
- `ga_qsvm/datasets/`: Runtime dataset registry and split preparation
- `ga_qsvm/runners/`: Runtime runner wiring
- `ga_qsvm/search/`: Hyperparameter search-space helpers
- `requirements.txt`: Project dependencies
- `qoop/`: Quantum Object Optimizer package

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project uses the QOOP (Quantum Object Optimizer) package developed by Vu Tuan Hai, Nguyen Tan Viet, and Le Bin Ho.
