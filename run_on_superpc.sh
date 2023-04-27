#!/bin/bash 
#SBATCH --job-name=neural_train
#SBATCH --time=0-0:15
#SBATCH --partition=gpu


echo "loading nvidia container"
module load singularity
singularity shell /opt/npad/shared/containers/nvhpc_22.5_devel.sif

echo "loading your .bashrc"
source ~/.bashrc

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

cargo r --release
