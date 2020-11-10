#!/bin/bash

module load gcc/7.3.1
module load cuda/10.1.243 

KOKKOS_DIR=/g/g16/jered/git/kokkos
NVCC_CXX=${KOKKOS_DIR}/bin/nvcc_wrapper

KOKKOS_INSTALL=/g/g16/jered/.local
CABANA_INSTALL=/g/g16/jered/.local
SILO_INSTALL=/g/g16/jered/.local

# Build ExaCLAMR on Lassen
rm -rf build
mkdir -p build
cd build

cmake -D CMAKE_CXX_FLAGS="-I${CABANA_INSTALL}/include -I${SILO_INSTALL}/include --expt-relaxed-constexpr" -D CMAKE_MODULE_PATH="${KOKKOS_INSTALL}/usr/local/lib64/cmake/Kokkos/" -D CMAKE_PREFIX_PATH="${KOKKOS_INSTALL}/usr/local;${SILO_INSTALL}/lib;${CABANA_INSTALL}/lib64/cmake/Cabana" -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_CUDA=ON ..
make -j8

mkdir data
mkdir data/raw

