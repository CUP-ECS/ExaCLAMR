/*
Authors: Patrick Bridges and Jered Dominguez-Trujillo
Description: Finite-Difference Solver of the Shallow Water Equations on a Regular Mesh Utilizing Kokkos, Cabana, and Cajita
Date: May 26, 2020
*/

#define DEBUG 0

#include <Solver.hpp>

#include <Cajita.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
    #include <stdio.h>
#endif

struct MeshInitFunc
{
    MeshInitFunc() {};

    template <typename state_t>
    KOKKOS_INLINE_FUNCTION
    bool operator()( const state_t r, const state_t rFill, state_t velocity[2], state_t &height ) const {
        velocity[0] = 0.0;
        velocity[1] = 0.0;
        height = ( r <= rFill ) ? 80.0 : 10.0;

        if ( DEBUG ) printf("r: %.4f\theight: %.4f\tvx: %.4f\tvy: %.4f\n", r, height, velocity[0], velocity[1]);

        return true;
    }
};


void clamr( const std::string& device,
            const std::array<double, 6>& global_bounding_box, 
            const std::array<int, 3>& global_num_cells, 
            const double rFill,
            const int halo_size, 
            const double gravity, 
            const double t_steps, 
            const int write_freq) {

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if ( rank == 0 ) {
        if( DEBUG ) printf( "ExaCLAMR\n" );
        if( DEBUG ) printf( "Nx: %d\tNy: %d\tNz: %d\tHalo Size: %d\t timesteps: %.4f\tGravity: %.4f\n", global_num_cells[0], global_num_cells[1], global_num_cells[2], halo_size, t_steps, gravity );
    }

    std::array<bool, 3> periodic = { false, false, false };

    std::array<int,3> ranks_per_dim = { 1, comm_size, 1 };
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    auto solver = ExaCLAMR::createSolver( device,
                                            MPI_COMM_WORLD, 
                                            MeshInitFunc(),
                                            global_bounding_box, 
                                            global_num_cells,
                                            rFill,
                                            periodic,
                                            partitioner, 
                                            halo_size, 
                                            t_steps, 
                                            gravity );

    solver->solve( write_freq );

}

int main( int argc, char* argv[] ) {

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    std::string device = "serial";

    int nx = 50, ny = 50, nz = 1;
    std::array<int, 3> global_num_cells = { nx, ny, nz };

    double hx = 50.0, hy = 50.0, hz = 1.0;
    std::array<double, 6> global_bounding_box = { 0, 0, 0, hx, hy, hz };

    double rFill = 6.0;

    int halo_size = 2;
    double gravity = 9.8;
    int t_steps = 2;
    int write_freq = 1;

    clamr( device,
            global_bounding_box, 
            global_num_cells, 
            rFill,
            halo_size,
            gravity, 
            t_steps, 
            write_freq );

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
