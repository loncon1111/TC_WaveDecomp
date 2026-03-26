#!/bin/bash

#SBATCH --job-name=decompose_vt
#SBATCH --time=10-00:00:00
#SBATCH -n 16
#SBATCH --output=decompose_vt.out
#SBATCH --error=decompose_vt.err

echo "Got $SLURM_NTASKS processors."

module load anaconda3/5.1.0_gnu_64
source activate env_clean        # adjust to your cluster's module name

python decompose_vt_parallel.py --cases -1 --workers $SLURM_NTASKS
