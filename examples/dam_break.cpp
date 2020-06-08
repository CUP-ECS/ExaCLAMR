/*
Authors: Patrick Bridges and Jered Dominguez-Trujillo
Description: Finite-Difference Solver of the Shallow Water Equations on a Regular Mesh Utilizing Kokkos, Cabana, and Cajita
Date: May 26, 2020
*/

#define DEBUG 0

#include <Input.hpp>
#include <Solver.hpp>

#include <Cajita.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>


#include <mpi.h>
#include <unistd.h>
#include <array>

#if DEBUG
    #include <iostream>
#endif

template <typename state_t>
struct MeshInitFunc
{
    
    state_t center[2];
    state_t width;

    MeshInitFunc(const std::array<double, 6>& box) {
	for ( int i = 0; i < 2; i++ ) {
		center[i] = (box[i+3] - box[i]) / 2.0;
	}
	width = box[3] - box[0];
    
	if ( DEBUG ) std::cout << "Center at " << center[0] << ", " << center[1] << "; total width is " << width << "\n";
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( const int coords[3], const state_t x[3], state_t velocity[2], state_t &height ) const {
        velocity[0] = 0.0;
        velocity[1] = 0.0;
        state_t r = sqrt( pow( x[0] - center[0], 2 ) + pow( x[1] - center[1], 2 ) );
        
        if ( DEBUG ) std::cout << x[0] << ", " << x[1] << " is " << r << " from the center: ";

        if ( r <= width * ( 6.0 / 128.0 ) ) {
            if ( DEBUG ) std::cout << "Tall\n";
            height = 100.0;
        } 
        else {
            if ( DEBUG ) std::cout << "Short\n";
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
    cl_args cl;

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    if ( parseInput( argc, argv, cl ) != 0 ) return -1;

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    std::string device = "openmp";

    if ( rank == 0 ) {
        // Print Results
        std::cout << "ExaClamr\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw(20) << "Cells"           << ": " << std::setw(8) << cl.nx << std::setw(8) << cl.ny << std::setw(8) << cl.nz << "\n";
        std::cout << std::left << std::setw(20) << "Domain"          << ": " << std::setw(8) << cl.hx << std::setw(8) << cl.hy << std::setw(8) << cl.hz << "\n";
        std::cout << std::left << std::setw(20) << "TimeSteps"       << ": " << std::setw(8) << cl.tSteps << "\n";
        std::cout << std::left << std::setw(20) << "Write Frequency" << ": " << std::setw(8) << cl.writeFreq << "\n";
        std::cout << "====================================\n";
    }

    std::array<int, 3> global_num_cells = { cl.nx, cl.ny, cl.nz };
    std::array<double, 6> global_bounding_box = { 0, 0, 0, cl.hx, cl.hy, cl.hz };

    double gravity = 9.8;
    

    clamr<double>( device,
            global_bounding_box, 
            global_num_cells, 
            cl.haloSize,
            gravity, 
            cl.tSteps, 
            cl.writeFreq );

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
