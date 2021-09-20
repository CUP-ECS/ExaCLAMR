#!/bin/bash

#PBS -q default
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:30:00
#PBS -n node-exclusive
#PBS -N ExaCLAMR
#PBS -j oe

module load openmpi-3.1.4-gcc-8.3.0-w3pkrvv

cd /users/jereddt/Stellar/ExaCLAMR

rm -rf data
mkdir -p data
mkdir -p data/raw

mpirun -np 8 ./build/examples/DamBreak
