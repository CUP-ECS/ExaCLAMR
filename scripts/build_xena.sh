#!/bin/bash

MYDIR=`pwd`
INSTALL_DIR=~/xena-scratch/tmp
NVCC_CXX=${MYDIR}/libs/kokkos/bin/nvcc_wrapper

module load cmake-3.13.4-gcc-7.4.0-rdapt4e
module load cuda-10.0.130-gcc-7.4.0-7svpedx
module load gcc-7.4.0-gcc-8.1.0-j26pfmd
module load openmpi-3.1.3-gcc-7.4.0-ts2kdgn

export NVCC_WRAPPER_DEFAULT_COMPILER=/opt/spack/opt/spack/linux-scientific7-x86_64/gcc-8.1.0/gcc-7.4.0-j26pfmdodybas2fpybqyi7hfvbc6kqot/bin/g++
export OMPI_CXX=${NVCC_CXX}

exit_abnormal() {
    echo "Error: Invalid option" >&2
    exit 1
}

build_kokkos() {
    cd ${MYDIR}/libs/kokkos
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_CXX_FLAGS="--expt-relaxed-constexpr" -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_CUDA_UVM=On -D Kokkos_ARCH_KEPLER35=On -D Kokkos_ENABLE_CUDA_LAMBDA=On -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
    make DESTDIR=${INSTALL_DIR} install -j8
    cd ${MYDIR}
}

build_cabana() {
    cd ${MYDIR}/libs/Cabana
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_PREFIX_PATH=${INSTALL_DIR}/usr/local -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
    make DESTDIR=${INSTALL_DIR} install -j8
    cd ${MYDIR}
}

update_repos() {
    cd ${MYDIR}
    if [ ! -d "libs" ]; then
        mkdir -p libs
        cd libs
        git clone https://github.com/kokkos/kokkos.git
        git clone https://github.com/ECP-cop/Cabana.git
        cd ${MYDIR}
    elif [ ! -d "libs/kokkos" ] || [ ! -d "libs/Cabana" ]; then
        if [ ! -d "libs/kokkos" ]; then
            cd libs
            git clone https://github.com/kokkos/kokkos.git
            cd ${MYDIR}
        fi
        if [ ! -d "libs/Cabana" ]; then
            cd libs
            git clone https://github.com/ECP-copa/Cabana.git
            cd ${MYDIR}
        fi
    else
        cd libs/kokkos
        git pull
        cd ${MYDIR}
        cd libs/Cabana
        git pull
        cd ${MYDIR}
    fi
}

build_workspace() {
    cd ${MYDIR}
    if [ ! -d "libs" ]; then
        mkdir -p libs
        cd libs
        git clone https://github.com/kokkos/kokkos.git
        build_kokkos
        cd ${MYDIR}/libs
        git clone https://github.com/ECP-copa/Cabana.git
        build_cabana
        cd ${MYDIR}
    else
        if [ ! -d "libs/kokkos" ]; then
            cd ${MYDIR}/libs
            git clone https://github.com/kokkos/kokkos.git
            build_kokkos
            cd ${MYDIR}
        fi
        if [ ! -d "libs/Cabana" ]; then
            cd ${MYDIR}/libs
            git clone https://github.com/ECP-copa/Cabana.git
            build_cabana
            cd ${MYDIR}
        fi
        cd ${MYDIR}
    fi
}

while getopts "a" opt; do
    case $opt in
        a)
            cd ${MYDIR}
            echo "Building and Installing all dependencies" >&2
            update_repos
            cd ${MYDIR}
            build_kokkos
            build_cabana
            ;;
        *)
            exit_abnormal
            ;;
    esac
done

build_workspace

cd ${MYDIR}
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_PREFIX_PATH=${INSTALL_DIR}/usr/local -D CMAKE_CXX_FLAGS="--expt-relaxed-constexpr" -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
make -j8

