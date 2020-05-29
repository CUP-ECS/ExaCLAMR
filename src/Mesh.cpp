#include <Mesh.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
template class Mesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<Kokkos::CudaSpace>;
#endif
} 
