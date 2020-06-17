#!/bin/bash

module load cmake-3.15.0-gcc-8.1.0-l3vpg4u
module load openmpi-3.1.3-gcc-8.1.0-gdwnwpj
module load gcc-8.1.0-gcc-4.8.5-3c5hjkq

cd ./libs/kokkos
rm -rf build
mkdir -p build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make DESTDIR=~/xena-scratch/tmp install
cd ../../Cabana/
rm -rf build
mkdir build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make DESTDIR=~/xena-scratch/tmp install
cd ../../../
rm -rf build
mkdir -p build
cd build
cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make
