#!/bin/bash
exit_abnormal() {                              # Function: Exit with error.
  echo "Error: Invalid option" >&2
  exit 1
}

while getopts "a" opt; do
    case $opt in
        a)
            echo "Building and Installing all dependencies" >&2
            cd libs
            cd Cabana
            git pull
            cd ..
            cd kokkos 
            git pull
            cd ..
            bash ./scripts/build.sh
            cd ..
            ;;
        *)
            exit_abnormal
            ;;
    esac
done

rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_C_COMPILER=/usr/local/Cellar/gcc/9.3.0_1/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9 -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
# cmake -D CMAKE_C_COMPILER=/usr/local/bin/gcc-8 -D CMAKE_CXX_COMPILER=/usr/local/bin/g++-8 -D Kokkos_ENABLE_OPENMP=On -D Kokkos_ENABLE_SERIAL=On ..
make -j 8
