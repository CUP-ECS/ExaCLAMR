#!/bin/bash

mpirun -np 1 -x OMP_PROC_BIND=spread -x OMP_NUM_THREADS=8 ./build/examples/DamBreak -n10 -h10 -r2.5