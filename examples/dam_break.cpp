/*
Authors: Patrick Bridges and Jered Dominguez-Trujillo
Description: Finite-Difference Solver of the Shallow Water Equations on a Regular Mesh Utilizing Kokkos, Cabana, and Cajita
Date: May 26, 2020
*/

#ifdef HAVE_SILO
    #include <silo.h>
#endif

#include <Solver.hpp>

#include <Cajita.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <array>

struct MeshInitFunc
{
    MeshInitFunc() {};

    template <typename state_t>
    KOKKOS_INLINE_FUNCTION
    bool operator()( const state_t x[3], state_t velocity[2], state_t &height ) const {
	velocity[0] = 0.0;
	velocity[1] = 0.0;
	if ( 1 <= x[0] && x[0] <= 3 &&
             1 <= x[1] && x[1] <= 3 )
        {
            height = 19.28077;
            printf("x: %.4f\ty: %.4f\tz: %.4f\n", x[0], x[1], x[2]);
        } else {
            height = 10.0;
        }

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

    if ( rank == 0 ) {
        printf( "CLAMR\n" );
        printf( "Nx: %d\tNy: %d\tNz: %d\tHalo Size: %d\t dt: %.4f\tGravity: %.4f\n", global_num_cells[0], global_num_cells[1], global_num_cells[2], halo_size, dt, gravity );
    }

    std::array<bool, 3> periodic = { false, false, false };

    std::array<int,3> ranks_per_dim = { 1, comm_size, 1 };
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    auto solver = ExaCLAMR::createSolver( device,
                                            MPI_COMM_WORLD, 
                                            MeshInitFunc(),
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

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    std::string device = "serial";

    double hx = 4.0, hy = 4.0, hz = 4.0;
    std::array<double, 6> global_bounding_box = { 0, 0, 0, hx, hy, hz };

    int nx = 4, ny = 4, nz = 1;
    std::array<int, 3> global_num_cells = { nx, ny, nz };

    int halo_size = 2;
    double dt = 0.034556;
    double gravity = 9.81;
    double t_final = 0.034556;
    int write_freq = 1;

    clamr( device,
            global_bounding_box, 
            global_num_cells, 
            halo_size,
            dt, 
            gravity, 
            t_final, 
            write_freq );

    #ifdef HAVE_SILO
        if ( rank == 0 ) {
            DBfile *silo_file;
            int		   driver = DB_PDB;

            DBShowErrors( DB_TOP, NULL );
            DBForceSingle( 1 );

            std::string s = "test.pdb";
            const char * filename = s.c_str();

            printf("Creating file: `%s'\n", filename);
            silo_file = DBCreate(filename, 0, DB_LOCAL, "Compound Array Test", driver);

            int sleepsecs = 10;
            int i = 1;

            if (sleepsecs)
                DBWrite (silo_file, "sleepsecs", &sleepsecs, &i, 1, DB_INT);

            DBClose(silo_file);
        }
    #endif

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
