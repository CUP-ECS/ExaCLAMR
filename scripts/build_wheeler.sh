#!/bin/bash

# Define Variables
MYDIR=`pwd`
INSTALL_DIR=~/wheeler-scratch/ExaCLAMR_Wheeler
KOKKOS_GIT=https://github.com/kokkos/kokkos.git
#KOKKOS_HASH=785d19f
CABANA_GIT=https://github.com/ECP-copa/Cabana.git
#CABANA_GIT=https://github.com/JDTruj2018/Cabana.git

# Load Modules
module load cmake-3.19.2-gcc-9.3.0-27lqgaf
module load openmpi-3.1.4-gcc-8.3.0-w3pkrvv
module load gcc-8.3.0-gcc-4.8.5-wwpinbr

# Abnormal Exit Message
exit_abnormal() {
    echo "Error: Invalid option" >&2
    exit 1
}

# Get Silo
get_silo() {
    cd ${MYDIR}/libs
    wget https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz
    tar -xvf silo-4.10.2.tgz
    mv silo-4.10.2 silo
    rm silo-4.10.2.tgz
    cd ${MYDIR}
}

build_silo() {
    cd ${MYDIR}/libs/silo
    ./configure --prefix=${INSTALL_DIR}
    make -j8 install
    cd ${MYDIR}
}

# Build Kokkos on Wheeler
build_kokkos() {
    cd ${MYDIR}/libs/kokkos
    #git checkout ${KOKKOS_HASH}
    rm -rf build
    mkdir -p build
    cd build
    cmake -D Kokkos_CXX_STANDARD=14 -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On  ..
    make DESTDIR=${INSTALL_DIR} install -j8
    cd ${MYDIR}
}

# Build Cabana on Wheeler
build_cabana() {
    cd ${MYDIR}/libs/Cabana
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_BUILD_TYPE="Debug" -D CMAKE_PREFIX_PATH="${INSTALL_DIR};${INSTALL_DIR}/usr/local" -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} -D Cabana_REQUIRE_OPNEMP=ON -D Cabana_ENABLE_EXAMPLES=ON -D Cabana_ENABLE_TESTING=ON -D Cabana_ENABLE_PERFORMANCE_TESTING=ON -D Cabana_ENABLE_CAJITA=ON -D Cabana_ENABLE_MPI=ON ..
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
        if [ ! -d "libs/silo" ]; then
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

# Build ExaCLAMR on Wheeler
cd ${MYDIR}
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_CXX_FLAGS=-I${INSTALL_DIR}/include -D CMAKE_PREFIX_PATH="${INSTALL_DIR}/usr/local;${INSTALL_DIR}/lib;${INSTALL_DIR}/lib64/cmake/Cabana" -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON ..
make -j8
mkdir data
mkdir data/raw
