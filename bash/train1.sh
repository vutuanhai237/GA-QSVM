cd /home/qsvm/GA-QSVM

conda activate cutn-qsvm

python3.10 main.py --num-circuit 8 --qubits 5 6 7 --data wine --num-generation 200

python3.10 main.py --num-circuit 8 --qubits 4 5 6 7 --data cancer --num-generation 200