#!/bin/bash

# Define Variables
MYDIR=`pwd`
INSTALL_DIR=~/xena-scratch/ExaCLAMR_Xena
NVCC_CXX=${MYDIR}/libs/kokkos/bin/nvcc_wrapper
KOKKOS_GIT=https://github.com/kokkos/kokkos.git
#KOKKOS_HASH=785d19f
CABANA_GIT=https://github.com/ECP-copa/Cabana.git
#CABANA_GIT=https://github.com/JDTruj2018/Cabana.git

# Load Modules
# module purge
module load gcc/9
module load cuda/11.0.2-lqua
module load cmake
module load openmpi

# Environment Variables for NVCC
export NVCC_WRAPPER_DEFAULT_COMPILER=/opt/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-9.3.0-oxjixak2sy5mudydbjatzvj4bpveh5jl/bin/gcc
export OMPI_CXX=${NVCC_CXX}

# Abnormal Exit Message
exit_abnormal() {
    echo "Error: Invalid option" >&2
    exit 1
}

# Get Silo
get_silo() {
    cd ${MYDIR}/libs
    wget https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2-bsd.tar.gz
    tar -xvf silo-4.10.2-bsd.tar.gz
    mv silo-4.10.2-bsd silo
    rm silo-4.10.2-bsd.tar.gz
    cd ${MYDIR}
}

# Build Silo
build_silo() {
    cd ${MYDIR}/libs/silo
    ./configure --prefix=${INSTALL_DIR}
    make -j8 install
    cd ${MYDIR}
}

# Build Kokkos on Xena (Tesla K40 GPU - KEPLER 35) with Cuda UVM
build_kokkos() {
    cd ${MYDIR}/libs/kokkos
    #git checkout ${KOKKOS_HASH}
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_CXX_FLAGS="--expt-relaxed-constexpr" -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_CUDA_LAMBDA=On -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On -D Kokkos_CXX_STANDARD=14 ..
    make DESTDIR=${INSTALL_DIR} install -j8
    cd ${MYDIR}
}

# Build Cabana on Xena (With Cuda)
build_cabana() {
    cd ${MYDIR}/libs/Cabana
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_BUILD_TYPE="Debug" -D CMAKE_PREFIX_PATH=${INSTALL_DIR}/usr/local -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Cabana_REQUIRE_OPNEMP=ON -D Cabana_ENABLE_EXAMPLES=ON -D Cabana_ENABLE_TESTING=ON -D Cabana_ENABLE_PERFORMANCE_TESTING=ON -D Cabana_ENABLE_CAJITA=On -D Cabana_ENABLE_MPI=On -D Cabana_REQUIRE_CUDA=On ..
    make install -j8
    cd ${MYDIR}
}

# Clone or Pull Dependencies into libs folder (Kokkos and Cabana)
update_repos() {
    cd ${MYDIR}
    if [ ! -d "libs" ]; then
        mkdir -p libs
        cd libs
        git clone ${KOKKOS_GIT}
        git clone ${CABANA_GIT}
        cd ${MYDIR}
    elif [ ! -d "libs/kokkos" ] || [ ! -d "libs/Cabana" ] || [ ! -d "libs/silo" ]; then
        if [ ! -d "libs/kokkos" ]; then
            cd libs
            git clone ${KOKKOS_GIT}
            cd ${MYDIR}
        fi
        if [ ! -d "libs/Cabana" ]; then
            cd libs
            git clone ${CABANA_GIT}
            cd ${MYDIR}
        fi
        if [ ! -d "libs/silo" ]; then
            cd libs
            get_silo
            cd ${MYDIR}
        fi
    else
        cd libs/kokkos
        git pull
        git checkout ${KOKKOS_HASH}
        cd ${MYDIR}
        cd libs/Cabana
        git pull
        cd ${MYDIR}
    fi
}

# Call Clone Build Functions as Needed
build_workspace() {
    cd ${MYDIR}
    if [ ! -d "libs" ]; then
        mkdir -p libs
        cd libs
        git clone ${KOKKOS_GIT}
        build_kokkos
        cd ${MYDIR}/libs
        git clone ${CABANA_GIT}
        build_cabana
        cd ${MYDIR}/libs
        get_silo
        build_silo
        cd ${MYDIR}
    else
        if [ ! -d "libs/kokkos" ]; then
            cd ${MYDIR}/libs
            git clone ${KOKKOS_GIT}
            build_kokkos
            cd ${MYDIR}
        fi
        if [ ! -d "libs/Cabana" ]; then
            cd ${MYDIR}/libs
            git clone ${CABANA_GIT}
            build_cabana
            cd ${MYDIR}
        fi
            if [ ! -d "libs/silo"  ]; then
            cd ${MYDIR}/libs
            get_silo
            build_silo
            cd ${MYDIR}
        fi
        cd ${MYDIR}
    fi
}

# Option if we want to Re-Pull and Re-Build Dependencies
while getopts "a" opt; do
    case $opt in
        a)
            cd ${MYDIR}
            echo "Building and Installing all dependencies" >&2
            update_repos
            cd ${MYDIR}
            build_kokkos
            build_cabana
            build_silo
            ;;
        *)
            exit_abnormal
            ;;
    esac
done

# Default Operation
build_workspace

# Build ExaCLAMR on Xena
cd ${MYDIR}
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_CXX_FLAGS=-I"${INSTALL_DIR}/include --expt-relaxed-constexpr"  -D CMAKE_PREFIX_PATH="${INSTALL_DIR}/usr/local;${INSTALL_DIR}/lib;${INSTALL_DIR}/lib64/cmake/Cabana" -D CMAKE_CXX_COMPILER=${NVCC_CXX} -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On -D Kokkos_ENABLE_CUDA=On ..
make -j8
mkdir data
mkdir data/raw

