#!/bin/bash 
#SBATCH --job-name=neural_train
#SBATCH --time=0-0:05
#SBATCH --partition=gpu


echo "loading nvidia container"
module load singularity
singularity shell /opt/npad/shared/containers/nvhpc_22.5_devel.sif

echo "loading your .bashrc"
source ~/.bashrc


cargo r --release
