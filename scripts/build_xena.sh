#!/bin/bash

#module load cmake-3.15.0-gcc-8.1.0-l3vpg4u
#module load openmpi-3.1.3-gcc-8.1.0-gdwnwpj
#module load cuda-10.0.130-gcc-8.1.0-3ledbtc
#module load gcc-8.1.0-gcc-4.8.5-3c5hjkq

module load cmake-3.13.4-gcc-7.4.0-rdapt4e
module load cuda-10.0.130-gcc-7.4.0-7svpedx
module load gcc-7.4.0-gcc-8.1.0-j26pfmd
module load openmpi-3.1.3-gcc-7.4.0-ts2kdgn
export NVCC_WRAPPER_DEFAULT_COMPILER=/opt/spack/opt/spack/linux-scientific7-x86_64/gcc-8.1.0/gcc-7.4.0-j26pfmdodybas2fpybqyi7hfvbc6kqot/bin/g++

#module load cmake-3.13.4-gcc-4.8.5-3mk5wsz
#module load cuda-10.0.130-gcc-4.8.5-p7udxbm
#module load openmpi-3.1.3-gcc-4.8.5-k6vjb53
#export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/lib64/ccache/g++

export OMPI_CXX=~/Research/LANL/ExaCLAMR/lib/kokkos/bin/nvcc_wrapper

cd ./libs/kokkos
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_CXX_FLAGS=--expt-relaxed-constexpr -D CMAKE_CXX_COMPILER=~/Research/LANL/ExaCLAMR/libs/kokkos/bin/nvcc_wrapper -D Kokkos_ENABLE_CUDA_UVM=On -D Kokkos_ARCH_KEPLER35=On -D Kokkos_ENABLE_CUDA_LAMBDA=On -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
make DESTDIR=~/xena-scratch/tmp install
cd ../../Cabana/
rm -rf build
mkdir build
cd build
cmake -D CMAKE_CXX_COMPILER=~/Research/LANL/ExaCLAMR/libs/kokkos/bin/nvcc_wrapper -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
make DESTDIR=~/xena-scratch/tmp install
cd ../../../
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_CXX_FLAGS=--expt-relaxed-constexpr -D CMAKE_CXX_FLAGS=--expt-extended-lambda -D CMAKE_CXX_COMPILER=~/Research/LANL/ExaCLAMR/libs/kokkos/bin/nvcc_wrapper -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
make
