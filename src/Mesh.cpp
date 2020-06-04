#include <Mesh.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
template class Mesh<Kokkos::HostSpace, Kokkos::Serial>;

#ifdef KOKKOS_ENABLE_OPENMP
template class Mesh<Kokkos::HostSpace, Kokkos::OpenMP>;
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<Kokkos::CudaSpace, Kokkos::Cuda>;
#endif
} 
