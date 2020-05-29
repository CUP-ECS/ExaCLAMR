set(HEADERS
  Solver.hpp
  Mesh.hpp
  ProblemManager.hpp
  )

set(SOURCES
  Mesh.cpp
  )

add_library(exaclamr ${SOURCES})

target_link_libraries(exaclamr
  Kokkos::kokkos
  MPI::MPI_CXX
  Cabana::cabanacore
  Cajita::Cajita )

target_include_directories(exaclamr
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
