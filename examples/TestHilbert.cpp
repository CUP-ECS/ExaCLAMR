#include <Kokkos_Core.hpp>
#include <Hilbert.hpp>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <mpi.h>

int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
    std::cout << "Testing Hilbert Layout\n"; 

    std::cout << "HilbertArray\n";
    Kokkos::View<double**, Kokkos::LayoutHilbert, Kokkos::HostSpace> HilbertArray( "Hilbert", 6, 4 );

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << "i: " << i << "\tj: " << j << "\tFlat Index: " << HilbertArray( i, j ) << "\n";
        }
    }

    auto s = Kokkos::subview( HilbertArray, 3, 3 );

    using cell_array = Cajita::Array<double, Cajita::Cell, Cajita::UniformMesh<double>, Kokkos::LayoutHilbert, Kokkos::HostSpace>;

    std::array<double, 3> global_low_corner;
    global_low_corner[0] = 0.0;
    global_low_corner[1] = 0.0;
    global_low_corner[2] = 0.0;

    std::array<double, 3> global_high_corner;
    global_high_corner[0] = 1.0;
    global_high_corner[1] = 1.0;
    global_high_corner[2] = 1.0;

    std::array<double, 3> cell_size;
    cell_size[0] = 0.25;
    cell_size[1] = 0.25;
    cell_size[2] = 0.25;

    Cajita::ManualPartitioner partitioner( { 1, 1, 1 } );

    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh, { false, false, false }, partitioner );
    auto local_grid = Cajita::createLocalGrid( global_grid, 2 );
    auto cell_scalar_layout = Cajita::createArrayLayout( local_grid, 1, Cajita::Cell() );

    std::shared_ptr<cell_array> height = Cajita::createArray<double, Kokkos::LayoutHilbert, Kokkos::HostSpace> ( "height", cell_scalar_layout );

    auto myview = height->view();

    for ( int i = 0; i < 4; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << myview( i, j, 0, 0 ) << "\n";
        }
    }
    
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
};