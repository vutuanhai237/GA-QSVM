#!/bin/bash
#PBS -N Digits_0
#PBS -o /home/nmduc/new_folder/output
#PBS -e /home/nmduc/new_folder/error
#PBS -l nodes=1:ppn=8
#PBS -q octa  #defines the destination queue of the job. 
cd /home/nmduc/new_folder/GA-QSVM/

module load python3.10

# Run the main script
python3.10 main.py --depth 4 --num-circuit 8 --qubits 3 4 5 6 7 8 9 10 --training-size 100 --test-size 50 --data digits