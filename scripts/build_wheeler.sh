#!/bin/bash

module load cmake-3.15.4-gcc-8.3.0-rmxifnl
module load openmpi-3.1.4-gcc-8.3.0-w3pkrvv
module load gcc-8.3.0-gcc-4.8.5-wwpinbr
cd ./libs/kokkos
rm -rf build
mkdir -p build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make DESTDIR=~/wheeler-scratch/tmp install
cd ../../Cabana/
rm -rf build
mkdir build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make DESTDIR=~/wheeler-scratch/tmp install
cd ../../../
rm -rf build
mkdir -p build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make
