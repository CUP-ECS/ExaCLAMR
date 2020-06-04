set(HEADERS
  Solver.hpp
  Mesh.hpp
  ProblemManager.hpp
  TimeIntegration.hpp
  )

set(SOURCES
  Mesh.cpp
  )

add_library(exaclamr ${SOURCES})

if (SILO)
  target_link_libraries(exaclamr
    Kokkos::kokkos
    MPI::MPI_CXX
    Cabana::cabanacore
    Cabana::Cajita
    ${SILO} )
else ()
  target_link_libraries(exaclamr
    Kokkos::kokkos
    MPI::MPI_CXX
    Cabana::cabanacore
    Cabana::Cajita
    )
endif()

target_include_directories(exaclamr
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
