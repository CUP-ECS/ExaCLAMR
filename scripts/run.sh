#!/bin/bash

rm -rf data
mkdir -p data
mkdir -p data/raw

mpirun -np 4 -x OMP_PROC_BIND=spread -x OMP_NUM_THREADS=8 ./build/examples/DamBreak -n500 -d500 -mopenmp