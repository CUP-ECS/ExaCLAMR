#!/bin/bash

# Define Variables
MYDIR=`pwd`
INSTALL_DIR=~/wheeler-scratch/tmp
KOKKOS_GIT=https://github.com/kokkos/kokkos.git
CABANA_GIT=https://github.com/ECP-copa/Cabana.git

# Load Modules
module load cmake-3.15.4-gcc-8.3.0-rmxifnl
module load openmpi-3.1.4-gcc-8.3.0-w3pkrvv
module load gcc-8.3.0-gcc-4.8.5-wwpinbr

# Abnormal Exit Message
exit_abnormal() {
    echo "Error: Invalid option" >&2
    exit 1
}

# Build Kokkos on Wheeler
build_kokkos() {
    cd ${MYDIR}/libs/kokkos
    rm -rf build
    mkdir -p build
    cd build
    cmake -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On  ..
    make DESTDIR=${INSTALL_DIR} install -j8
    cd ${MYDIR}
}

# Build Cabana on Wheeler
build_cabana() {
    cd ${MYDIR}/libs/Cabana
    rm -rf build
    mkdir -p build
    cd build
    cmake -D CMAKE_PREFIX_PATH=${INSTALL_DIR}/usr/local -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
    make DESTDIR=${INSTALL_DIR} install -j8
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
    elif [ ! -d "libs/kokkos" ] || [ ! -d "libs/Cabana" ]; then
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
    else
        cd libs/kokkos
        git pull
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
cmake -D CMAKE_PREFIX_PATH=${INSTALL_DIR}/usr/local -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make -j8
