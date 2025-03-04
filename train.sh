#!/bin/bash
#PBS -N GA-QSVM_2
#PBS -o /home/nmduc/GA-QSVM/output
#PBS -e /home/nmduc/GA-QSVM/error
#PBS -l nodes=1:ppn=20
#PBS -q parallel  #defines the destination queue of the job. 
cd /home/nmduc/GA-QSVM
module load python3.10

# Activate the Python virtual environment
source ga-qsvm/bin/activate

# Run the main script
python3.10 main.py

# Deactivate the virtual environment when done
deactivate
