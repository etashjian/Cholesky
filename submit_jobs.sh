#!/bin/bash

#SBATCH -p slurm_me759
#SBATCH --job-name=cholesky
#SBATCH -N 1 -n 16 --gres=gpu:1
#SBATCH -o test/grid_output.txt

#./build/bin/simple_band 4096 128
#./build/bin/mat_band_compare
#./build/bin/mat_dim_compare
./build/bin/mat_compare
