#!/bin/bash

#BSUB -nnodes 4
#BSUB -W 60
#BSUB -G unm
#BSUB -J ExaCLAMR_Test
#BSUB -q pbatch

module load gcc/7.3.1
module load cuda/10.1.243

cd /g/g16/jered/CUP-ECS/ExaCLAMR

rm -rf data
mkdir -p data
mkdir -p data/raw

lrun -N 1 -T 1 -x OMP_PROC_BIND=spread ./build/examples/DamBreak -n100 -d100
lrun -N 1 -T 1 -x OMP_PROC_BIND=spread -x OMP_NUM_THREADS=10 ./build/examples/DamBreak -n100 -d100 -mopenmp
lrun -N 1 -T 1 ./build/examples/DamBreak -n1000 -d1000 -mcuda

lrun -N 4 -T 1 -x OMP_PROC_BIND=spread ./build/examples/DamBreak -n1000 -d1000 -mserial
lrun -N 4 -T 1 -x OMP_PROC_BIND=spread -x OMP_NUM_THREADS=10 ./build/examples/DamBreak -n1000 -d1000 -mopenmp
lrun -N 4 -T 1 ./build/examples/DamBreak -n1000 -d1000 -mcuda

