/*
Authors: Patrick Bridges and Jered Dominguez-Trujillo
Description: Finite-Difference Solver of the Shallow Water Equations on a Regular Mesh Utilizing Kokkos, Cabana, and Cajita
Date: May 26, 2020
*/

#include <Solver.hpp>

#include <Cajita.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>
#include <stdio.h>
#include <array>

struct ClamrInitFunc
{
    double _test;

    ClamrInitFunc()
    {};

    template<class CellType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( const double momentum[2], const double height, CellType& c ) const {

        for( int dim = 0; dim < 2; ++dim ) Cabana::get<0>( c, dim ) = 0.0;
        Cabana::get<1>( c ) = 1.0;

        return true;
    }
};


void clamr( const std::string& device,
            const std::array<double, 6>& global_bounding_box, 
            const std::array<int, 3>& global_num_cells, 
            const int halo_size, 
            const double dt, 
            const double gravity, 
            const double t_final, 
            const int write_freq) {

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    printf( "CLAMR\n" );
    printf( "Nx: %d\tNy: %d\tNz: %d\tHalo Size: %.4f\t dt: %.4f\tGravity: %.4f\n", global_num_cells[0], global_num_cells[1], global_num_cells[2], halo_size, dt, gravity );


    std::array<bool, 3> periodic = { false, false, false };

    std::array<int,3> ranks_per_dim = { 1, comm_size, 1 };
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    auto solver = ExaCLAMR::createSolver( device,
                                            MPI_COMM_WORLD, 
                                            ClamrInitFunc(),
                                            global_bounding_box, 
                                            global_num_cells,
                                            periodic,
                                            partitioner, 
                                            halo_size, 
                                            dt, 
                                            gravity );
    solver->solve( t_final, write_freq );

}

int main( int argc, char* argv[] ) {

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    std::string device = "serial";

    double hx = 1.0, hy = 1.0, hz = 0.25;
    std::array<double, 6> global_bounding_box = { 0, 0, 0, hx, hy, hz };

    int nx = 4, ny = 4, nz = 1;
    std::array<int, 3> global_num_cells = { nx, ny, nz };

    int halo_size = 2;
    double dt = 0.1;
    double gravity = 10.0;
    double t_final = 40.0;
    int write_freq = 10;

    clamr( device,
            global_bounding_box, 
            global_num_cells, 
            halo_size,
            dt, 
            gravity, 
            t_final, 
            write_freq );

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}