#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodelist=a6000

# Place your command(s) here:
# Conda setup
__conda_setup="$('/home/qsvm/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/qsvm/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/qsvm/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/qsvm/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# conda setup done
conda activate cutn-qsvm

# python benchmark.py --num-generation 50 100 150 200 250
python benchmark.py --num-circuit 4 8 16 20
python benchmark.py --prob-mutate 0.001, 0.01, 0.1, 0.3, 0.5
