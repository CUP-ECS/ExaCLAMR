#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>


void haloExchange() {
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    std::array<bool, 3> periodic = { false, false, false };


    // 2-D Mesh - Z Domain is From 0 to 1
    std::array<double, 6> bounding_box;
    bounding_box[0] = 0;   // Set X Min Coordinate
    bounding_box[1] = 0;   // Set Y Min Coordinate
    bounding_box[2] = 0;   // Set Z Min Coordinate
    bounding_box[3] = 4; // Set X Max Coordinate
    bounding_box[4] = 4; // Set Y Max Coordinate
    bounding_box[5] = 4; // Set Z Max Coordinate

    // 3-D Mesh
    std::array<int, 3> num_cell;
    num_cell[0] = 4; // Set X Num Cells
    num_cell[1] = 4; // Set Y Num Cells
    num_cell[2] = 4; // Set Z Num Cells

    // Calculate Cell Size
    std::array<double, 3> cell_size;
    cell_size[0] = ( bounding_box[3] - bounding_box[0] ) / num_cell[0]; // X Cell Size
    cell_size[1] = ( bounding_box[4] - bounding_box[1] ) / num_cell[1]; // Y Cell Size
    cell_size[2] = ( bounding_box[5] - bounding_box[2] ) / num_cell[2]; // Z Cell Size

    // Set Lower and Upper Corner Array
    std::array<double, 3> global_low_corner  = { bounding_box[0], bounding_box[1], bounding_box[2] };
    std::array<double, 3> global_high_corner = { bounding_box[3], bounding_box[4], bounding_box[5] };

    // Create Global Mesh and Global Grid - Need to Create Local Grid
    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Array parameters.
    unsigned halo_width = 2;

    // Create a cell array.
    auto layout = Cajita::createArrayLayout( global_grid, halo_width, 1, Cajita::Cell() );
    auto array = Cajita::createArray<double, DeviceType>( "array", layout );

    // Assign the owned cells a value of 1 and the rest 0.
    Cajita::ArrayOp::assign( *array, rank, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *array, rank, Cajita::Own() );

    auto owned_space = array->layout()->indexSpace( Cajita::Own(), Cajita::Local() );
    auto ghosted_space = array->layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), array->view() );

    if ( rank == 0 ) { 
        for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
            for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
                for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
                    for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                        if ( ! ( i < owned_space.min( Cajita::Dim::I ) - halo_width ||
                            i >= owned_space.max( Cajita::Dim::I ) + halo_width ||
                            j < owned_space.min( Cajita::Dim::J ) - halo_width ||
                            j >= owned_space.max( Cajita::Dim::J ) + halo_width ||
                            k < owned_space.min( Cajita::Dim::K ) - halo_width ||
                            k >= owned_space.max( Cajita::Dim::K ) + halo_width ) )
                                std::cout << "(" << i << ", " << j << ", " << k << ", " << l << "): " << host_view( i, j, k, l ) << std::endl;
    }

    // Create a halo.
    auto halo = createHalo( *array, Cajita::FullHaloPattern(), halo_width );

    // Gather into the ghosts.
    halo->gather( ExecutionSpace(), *array );


    if ( rank == 0 ) { 
        for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
            for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
                for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
                    for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                        if ( ! ( i < owned_space.min( Cajita::Dim::I ) - halo_width ||
                            i >= owned_space.max( Cajita::Dim::I ) + halo_width ||
                            j < owned_space.min( Cajita::Dim::J ) - halo_width ||
                            j >= owned_space.max( Cajita::Dim::J ) + halo_width ||
                            k < owned_space.min( Cajita::Dim::K ) - halo_width ||
                            k >= owned_space.max( Cajita::Dim::K ) + halo_width ) )
                                std::cout << "(" << i << ", " << j << ", " << k << ", " << l << "): " << host_view( i, j, k, l ) << std::endl;
    }

}

int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    haloExchange();

    MPI_Finalize();

    return 0;
}