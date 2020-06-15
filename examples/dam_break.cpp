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
void clamr(  cl_args<state_t>& cl, ExaCLAMR::Timer& timer) {
    int comm_size, rank;                                        // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );                // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );                     // Get My Rank

    timer.setupStart();

    // Splits Ranks up as Evenly as Possible Across X and Y Dimensions
    int x_ranks = comm_size;
    while ( x_ranks % 2 == 0 && x_ranks > 2 ) {
        x_ranks /= 2;
    }
    int y_ranks = comm_size / x_ranks;
    if ( DEBUG ) std::cout << "X Ranks: " << x_ranks << " Y Ranks: " << y_ranks << "\n";

    std::array<int,3> ranks_per_dim = { x_ranks, y_ranks, 1 };      // Ranks per Dimension

    Cajita::ManualPartitioner partitioner( ranks_per_dim );     // Create Cajita Partitioner

    // Create Solver
    auto solver = ExaCLAMR::createSolver( cl,
                                        MPI_COMM_WORLD, 
                                        MeshInitFunc<state_t>( cl.global_bounding_box ),
                                        partitioner, 
                                        timer );
    timer.setupStop();

    // Solve
    solver->solve( cl.write_freq, timer );
};


int main( int argc, char* argv[] ) {
    // Initialize and Start Timer
    ExaCLAMR::Timer timer( ExaCLAMR::TimerType::AGGREGATE );
    timer.overallStart();

    timer.setupStart();
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

    timer.setupStop();

    timer.writeStart();
    // Only Rank 0 Print Command Line Options
    if ( rank == 0 ) {
        // Print Command Line Options
        std::cout << "ExaClamr\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"    << ": " << std::setw( 8 ) << cl.device      << "\n";                                                           // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Cells"             << ": " << std::setw( 8 ) << cl.nx          << std::setw( 8 ) << cl.ny << std::setw( 8 ) << cl.nz << "\n";     // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Domain"            << ": " << std::setw( 8 ) << cl.hx          << std::setw( 8 ) << cl.hy << std::setw( 8 ) << cl.hz << "\n";     // Span of Domain
        std::cout << std::left << std::setw( 20 ) << "Periodicity"       << ": " << std::setw( 8 ) <<                                                                                   // Periodicity
        cl.periodic[0] << std::setw( 8 ) << cl.periodic[1] << std::setw( 8 ) << cl.periodic[2] << "\n";
        std::cout << std::left << std::setw( 20 ) << "Sigma"             << ": " << std::setw( 8 ) << cl.sigma       << "\n";                                                           // Sigma
        std::cout << std::left << std::setw( 20 ) << "Gravity"           << ": " << std::setw( 8 ) << cl.gravity     << "\n";                                                           // Gravitational Constant
        std::cout << std::left << std::setw( 20 ) << "Time Steps"        << ": " << std::setw( 8 ) << cl.time_steps  << "\n";                                                           // Number of Time Steps
        std::cout << std::left << std::setw( 20 ) << "Write Frequency"   << ": " << std::setw( 8 ) << cl.write_freq  << "\n";                                                           // Time Steps between each Write
        std::cout << "====================================\n";
    }
    timer.writeStop();

    // Call Clamr - Double or Float as Template Arg
    clamr<state_t>( cl, timer );

    Kokkos::finalize();                                 // Finalize Kokkos
    MPI_Finalize();                                     // Finalize MPI

    timer.overallStop();

    if (rank == 0 ) {
        timer.report();
    }

    return 0;
};
