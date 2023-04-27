#!/bin/bash 
#SBATCH --job-name=neural_train
#SBATCH --time=0-0:30
#SBATCH --partition=gpu

module load singularity
singularity shell /opt/npad/shared/containers/nvhpc_22.5_devel.sif
source ~/.bashrc

cargo r --release
