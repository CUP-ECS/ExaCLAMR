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
#include <unistd.h>
#include <array>
#include <iostream>

template <typename state_t>
struct MeshInitFunc
{
    
    state_t center[2];
    state_t width;

    MeshInitFunc(const std::array<double, 6>& box) {
	for (int i = 0; i < 2; i++) {
		center[i] = (box[i+3] - box[i]) / 2.0;
	}
	width = box[3] - box[0];
	std::cout << "Center at " << center[0] << ", " << center[1] << "; total width is " << width << "\n";
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( const int coords[3], const state_t x[3], state_t velocity[2], state_t &height ) const {
	velocity[0] = 0.0;
	velocity[1] = 0.0;
	state_t r = sqrt( pow( x[0] - center[0], 2 ) + pow( x[1] - center[1], 2 ));
	std::cout << x[0] << ", " << x[1] << " is " << r << " from the center: ";
	if ( r <= width*(6.0/128.0) )
        {
	    std::cout << "Tall\n";
            height = 100.0;
        } else {
	    std::cout << "Short\n";
            height = 7.0;
        }

        return true;
    }
};


template <typename state_t>
void clamr( const std::string& device,
            const std::array<double, 6>& global_bounding_box, 
            const std::array<int, 3>& global_num_cells, 
            const int halo_size, 
            const double gravity, 
            const double t_steps, 
            const int write_freq) {

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if ( rank == 0 ) {
        printf( "CLAMR\n" );
        printf( "Nx: %d\tNy: %d\tNz: %d\tHalo Size: %d\t timesteps: %.4f\tGravity: %.4f\n", global_num_cells[0], global_num_cells[1], global_num_cells[2], halo_size, t_steps, gravity );
    }

    std::array<bool, 3> periodic = { false, false, false };

    std::array<int,3> ranks_per_dim = { 1, comm_size, 1 };
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    auto solver = ExaCLAMR::createSolver( device,
                                            MPI_COMM_WORLD, 
                                            MeshInitFunc<state_t>(global_bounding_box),
                                            global_bounding_box, 
                                            global_num_cells,
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

    double hx = double( nx ), hy = double( ny ), hz = 1.0;
    std::array<double, 6> global_bounding_box = { 0, 0, 0, hx, hy, hz };

    int halo_size = 2;
    double gravity = 9.8;
    int t_steps = 10;
    int write_freq = 1;

    clamr<double>( device,
            global_bounding_box, 
            global_num_cells, 
            halo_size,
            gravity, 
            t_steps, 
            write_freq );

    #ifdef HAVE_SILO
        if ( rank == 0 ) {
            DBfile *silo_file;
            int		   driver = DB_PDB;

            DBShowErrors( DB_TOP, NULL );
            DBForceSingle( 1 );

            std::string s = "test.pdb";
            const char * filename = s.c_str();

            std::cout << "Creating file: `" << filename << "'\n";
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
