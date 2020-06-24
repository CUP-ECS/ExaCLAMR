# project settings
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

project(ExaCLAMR LANGUAGES CXX VERSION 0.1.0)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_library(SILO NAMES silo libsilo libsilo.a HINTS "/usr/local/lib")

if (SILO)
    message(STATUS "SILO: [${SILO}]")
    add_definitions(-DHAVE_SILO)
else ()
    message(STATUS "Did not find Silo")
endif()

find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
find_package(Cabana REQUIRED COMPONENTS Cabana::Cajita Cabana::cabanacore)

include_directories(include)

add_subdirectory(src)

add_subdirectory(examples)
