#!/bin/bash
#SBATCH --job-name=ExaCLAMR # Job name
#SBATCH --nodes=2                    # Run all processes on a single node
#SBATCH --ntasks=2                   # Number of processes
#SBATCH --gpus=2                     # GPUs
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --output=multiprocess_%j.log # Standard output and error log

module purge
module load gcc/8.3.0-wbma
module load openmpi/4.0.5-cuda-rla7
module load cmake
module load cuda/11.2.0-qj6z

mkdir -p data
mkdir -p data/raw

rm -rf data/raw/*

srun -N 2 -n 2 ./build/examples/DamBreak -mcuda -n1000 -t100 -w10

