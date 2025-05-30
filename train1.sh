#!/bin/bash
#PBS -N Digits_1
#PBS -o /home/nmduc/GA-QSVM/output
#PBS -e /home/nmduc/GA-QSVM/error
#PBS -l nodes=1:ppn=8
#PBS -q octa  #defines the destination queue of the job. 
cd /home/nmduc/GA-QSVM
module load python3.10

# Activate the Python virtual environment
source ga-qsvm/bin/activate

# Run the main script
python3.10 main.py --depth 4 --num-circuit 8 --qubits 3 4 5 --num-machines 3 --id 1 --training-size 300 --test-size 0 --data digits


# Deactivate the virtual environment when done
deactivate
