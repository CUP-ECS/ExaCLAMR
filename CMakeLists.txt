# project settings
cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 14)

project(ExaCLAMR LANGUAGES CXX VERSION 0.1.0)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
find_package(Cabana REQUIRED)
find_package(Cajita REQUIRED)

include_directories(include)
add_subdirectory(src)

add_subdirectory(examples)
