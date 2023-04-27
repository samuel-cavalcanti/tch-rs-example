#!/bin/bash 
#SBATCH --job-name=neural_train
#SBATCH --time=0-0:15
#SBATCH --partition=gpu
#SBATCH --exclusive

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

cargo r --release
