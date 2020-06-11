/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef DEBUG
    #define DEBUG 0 
#endif

#include <Input.hpp>
#include <Solver.hpp>
#include <Timer.hpp>

#include <Cajita.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
    #include <iostream>
#endif

#define MICROSECONDS 1.0e-6


template <typename state_t>
struct MeshInitFunc
{
    // Initialize Variables
    state_t center[2], width[3];

    MeshInitFunc( const std::array<state_t, 6>& box ) {
        // Calculate X, Y, Z, Center and Width
        for ( int i = 0; i < 2; i++ ) {
            width[i] = box[i + 3] - box[i];
            center[i] = box[i + 3] - ( ( width[i] ) / 2.0 );
        }
        
        // DEBUG: Print 2-D Center Location and X, Y Width
        if ( DEBUG ) std::cout << "Center: (" << center[0] << ", " << center[1] << \
        ")\tX-width: " << width[0] << "\tY-width: " << width[1] << "\n";       
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( const int coords[3], const state_t x[3], state_t velocity[2], state_t &height ) const {
        // Initialize Velocity
        velocity[0] = 0.0, velocity[1] = 0.0;       // Vx = 0.0, Vy = 0.0

        // Calculate Distance from the Center
        state_t r = sqrt( pow( x[0] - center[0], 2 ) + pow( x[1] - center[1], 2 ) );
        
        // DEBUG: Print (X, Y) Coordinate and Distance from the Center
        if ( DEBUG ) std::cout << x[0] << ", " << x[1] << " is " << r << " from the center: ";

        // Set Height
        // TODO: Make it Possible to Tweak Tall and Short Values, Along with R Threshold Calculation with Command Line Args
        if ( r <= width[0] * ( 6.0 / 128.0 ) ) {
            // DEBUG: Print Indicated Tall Height Assigned to Point
            if ( DEBUG ) std::cout << "Tall\n";
            height = 100.0;
        } 
        else {
            // DEBUG: Print Inidicating Short Height Assigned to Point
            if ( DEBUG ) std::cout << "Short\n";
            height = 7.0;
        }

        return true;
    };
};


template <typename state_t>
void clamr( const std::string& device,
            const std::array<state_t, 6>& global_bounding_box, 
            const std::array<int, 3>& global_num_cells,
            const int halo_size, 
            const int time_steps,
            const state_t gravity,
            const int write_freq) {
    int comm_size, rank;                                        // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );                // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );                     // Get My Rank

    // TODO: Move to main and Possibly ParseInput
    std::array<bool, 3> periodic = { false, false, false };     // Not Periodic for the Moment

    // TODO: Generalize
    std::array<int,3> ranks_per_dim = { 1, comm_size, 1 };      // Ranks per Dimension

    Cajita::ManualPartitioner partitioner( ranks_per_dim );     // Create Cajita Partitioner

    // Create Solver
    auto solver = ExaCLAMR::createSolver( device,
                                            MPI_COMM_WORLD, 
                                            MeshInitFunc<state_t>( global_bounding_box ),
                                            global_bounding_box, 
                                            global_num_cells,
                                            periodic,
                                            partitioner, 
                                            halo_size, 
                                            time_steps, 
                                            gravity );

    // Solve
    solver->solve( write_freq );

    state_t max_compute, max_communicate;

    // Compute Time
    state_t time_compute = solver->timeCompute();
    // Communicate Time
    state_t time_communicate = solver->timeCommunicate();

    // TODO: Scenario where we need MPI_FLOAT
    MPI_Allreduce( &time_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD ); 

    // TODO: Scenario where we need MPI_FLOAT
    MPI_Allreduce( &time_communicate, &max_communicate, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    if (rank == 0 ) {
        std::cout << "The Total Compute Time of the Program was " << max_compute << " seconds\n";
        std::cout << "The Total Communication Time of the Program was " << max_communicate << " seconds\n";
    }
};


int main( int argc, char* argv[] ) {
    // Initialize and Start Timer
    using timestruct = std::chrono::high_resolution_clock::time_point;
    timestruct timer_total;
    Timer::timer_start( &timer_total );                  // Start Timer

    // Using doubles
    using state_t = double;

    MPI_Init( &argc, &argv );                           // Initialize MPI
    Kokkos::initialize( argc, argv );                   // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );        // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );             // My Rank

    // Parse Input
    cl_args<state_t> cl;
    if ( parseInput( rank, argc, argv, cl ) != 0 ) return -1; // Return if Failed

    // TODO: Add to parseInput
    // TODO: Add Cuda
    std::string device = "openmp";                      // serial or openmp
    state_t gravity = 9.8;                              // Gravitational Constant ( m / s^2 )

    // Only Rank 0 Print Command Line Options
    if ( rank == 0 ) {
        // Print Command Line Options
        std::cout << "ExaClamr\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Cells"           << ": " << std::setw( 8 ) << cl.nx << std::setw( 8 ) << cl.ny << std::setw( 8 ) << cl.nz << "\n";    // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Domain"          << ": " << std::setw( 8 ) << cl.hx << std::setw( 8 ) << cl.hy << std::setw( 8 ) << cl.hz << "\n";    // Span of Domain ( In Meters )
        std::cout << std::left << std::setw( 20 ) << "TimeSteps"       << ": " << std::setw( 8 ) << cl.time_steps << "\n";                                                  // Number of Time Steps
        std::cout << std::left << std::setw( 20 ) << "Write Frequency" << ": " << std::setw( 8 ) << cl.write_freq << "\n";                                                  // Time Steps between each Write
        std::cout << "====================================\n";
    }

    // Create Cell and Bounding Box Arrays
    std::array<int, 3> global_num_cells = { cl.nx, cl.ny, cl.nz };
    std::array<state_t, 6> global_bounding_box = { 0, 0, 0, cl.hx, cl.hy, cl.hz };

    // Call Clamr - Double or Float as Template Arg
    clamr<state_t>( device,
            global_bounding_box, 
            global_num_cells, 
            cl.halo_size,
            cl.time_steps,
            gravity,  
            cl.write_freq );

    Kokkos::finalize();                                 // Finalize Kokkos                                  
    MPI_Finalize();                                     // Finalize MPI

    if (rank == 0 ) {
        state_t time_total = ( state_t ) Timer::timer_stop( timer_total ) * MICROSECONDS;           // Stop Timer
        std::cout << "The Total Execution Time of the Program was " << time_total << " seconds\n";  // Print Total Program Time
    }

    return 0;
};
