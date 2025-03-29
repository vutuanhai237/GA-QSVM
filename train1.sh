#!/bin/bash
#PBS -N GA-QSVM_1
#PBS -o /media/data/nmduc/GA-QSVM/output
#PBS -e /media/data/nmduc/GA-QSVM/error
#PBS -l nodes=1:ppn=8
#PBS -q octa  #defines the destination queue of the job. 
cd /media/data/nmduc/GA-QSVM
module load python3.10

# Activate the Python virtual environment
source ga-qsvm/bin/activate

# Run the main script
python3.10 main.py --depth 4 --num-circuit 8 --qubits 3 4 5 --num-machines 3 --id 1 --start-index 142

# Deactivate the virtual environment when done
deactivate
