#!/bin/bash

#SBATCH -p slurm_me759
#SBATCH --job-name=cholesky
#SBATCH -N 1 -n 1 --gres=gpu:1
#SBATCH -o test/grid_output.txt

./bin/simple_band 2000 80
