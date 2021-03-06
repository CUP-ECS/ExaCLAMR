/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Dam Break Problem on a Regular Grid using the Shallow Water Equations:
 * Cells Inside a Circular Radius of Size ( 6 / 128 ) * Width, Height = 80
 * Cells Inside a Circular Radius of Size 1.5 * ( 6 / 128 ) * Width and outside of Inner Circle, Height = 70
 * Other Cells, Height = 10
 * All Cells, X-Momentum = 0, Y-Momentum = 0
 */

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <BoundaryConditions.hpp>
#include <ExaClamrTypes.hpp>
#include <Input.hpp>
#include <Solver.hpp>
#include <Timer.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
#include <iostream>
#endif

// Initialization Function
template <typename state_t>
struct MeshInitFunc {
    // Initialize Variables
    state_t center[2], width[3];

    MeshInitFunc( const std::array<state_t, 6> &box ) {
        // Calculate X, Y, Z, Center and Width
        for ( int i = 0; i < 2; i++ ) {
            width[i]  = box[i + 3] - box[i];
            center[i] = box[i + 3] - ( ( width[i] ) / 2.0 );
        }

        // DEBUG: Print 2-D Center Location and X, Y Width
        // if ( DEBUG ) std::cout << "Center: (" << center[0] << ", " << center[1] << \
        ")\tX-width: " << width[0] << "\tY-width: " << width[1] << "\n";
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( const int coords[3], const state_t x[3], state_t velocity[2], state_t &height ) const {
        // Initialize Velocity
        velocity[0] = 0.0, velocity[1] = 0.0; // Vx = 0.0, Vy = 0.0

        // Calculate Distance from the Center
        state_t r = sqrt( pow( x[0] - center[0], 2 ) + pow( x[1] - center[1], 2 ) );

        // DEBUG: Print (X, Y) Coordinate and Distance from the Center
        // if ( DEBUG ) std::cout << x[0] << ", " << x[1] << " is " << r << " from the center: ";

        state_t threshold = width[0] * ( 6.0 / 128.0 );

        // Set Height
        if ( r <= threshold ) {
            // DEBUG: Print Indicated Tall Height Assigned to Point
            // if ( DEBUG ) std::cout << "Tall\n";
            height = 80.0;
        } else if ( r <= 1.5 * threshold ) {
            height = 70.0;
        } else {
            // DEBUG: Print Indicating Short Height Assigned to Point
            // if ( DEBUG ) std::cout << "Short\n";
            height = 10.0;
        }

        return true;
    };
};

// Create Solver and Run CLAMR
template <typename state_t>
void clamr( ExaCLAMR::ClArgs<state_t> &cl, ExaCLAMR::BoundaryCondition &bc, ExaCLAMR::Timer &timer ) {
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    timer.setupStart();
    // Splits Ranks up as Evenly as Possible Across X and Y Dimensions
    int x_ranks = comm_size;
    while ( x_ranks % 2 == 0 && x_ranks > 2 ) {
        x_ranks /= 2;
    }
    int y_ranks = comm_size / x_ranks;
    if ( DEBUG ) std::cout << "X Ranks: " << x_ranks << " Y Ranks: " << y_ranks << "\n";

    std::array<int, 3> ranks_per_dim = { x_ranks, y_ranks, 1 }; // Ranks per Dimension

    Cajita::ManualPartitioner partitioner( ranks_per_dim ); // Create Cajita Partitioner

    // Create Solver
    if ( !cl.meshtype.compare( "regular" ) ) {
        auto solver = ExaCLAMR::createRegularSolver( cl, bc, MPI_COMM_WORLD, MeshInitFunc<state_t>( cl.global_bounding_box ), partitioner, timer );
        timer.setupStop();
        // Solve
        solver->solve( cl.write_freq, timer );
    } else if ( !cl.meshtype.compare( "amr" ) )
        auto solver = ExaCLAMR::createAMRSolver( cl, bc, MPI_COMM_WORLD, MeshInitFunc<state_t>( cl.global_bounding_box ), partitioner, timer );
    else
        auto solver = ExaCLAMR::createRegularSolver( cl, bc, MPI_COMM_WORLD, MeshInitFunc<state_t>( cl.global_bounding_box ), partitioner, timer );
};

int main( int argc, char *argv[] ) {
    // Initialize and Start Timer
    ExaCLAMR::Timer timer( ExaCLAMR::TimerType::AGGREGATE );
    timer.overallStart();

    timer.setupStart();

    // Using doubles
    using state_t = double;

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    // Parse Input
    ExaCLAMR::ClArgs<state_t> cl;
    if ( ExaCLAMR::parseInput( rank, argc, argv, cl ) != 0 ) return -1;

    // Define boundary conditions to be Reflective in 2-Dimensions
    ExaCLAMR::BoundaryCondition bc;
    bc.boundary_type[0] = ExaCLAMR::BoundaryType::REFLECTIVE; // X - Left
    bc.boundary_type[1] = ExaCLAMR::BoundaryType::REFLECTIVE; // Y - Bottom
    bc.boundary_type[2] = ExaCLAMR::BoundaryType::NONE;       // Z -
    bc.boundary_type[3] = ExaCLAMR::BoundaryType::REFLECTIVE; // X - Right
    bc.boundary_type[4] = ExaCLAMR::BoundaryType::REFLECTIVE; // Y - Top
    bc.boundary_type[5] = ExaCLAMR::BoundaryType::NONE;       // Z -

    timer.setupStop();

    timer.writeStart();
    // Only Rank 0 Prints Command Line Options
    if ( rank == 0 ) {
        // Print Command Line Options
        std::cout << "ExaClamr\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.device << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Mesh Type"
                  << ": " << std::setw( 8 ) << cl.meshtype << "\n"; // Mesh Type
        /*
        std::cout << std::left << std::setw( 20 ) << "Ordering"
                  << ": " << std::setw( 8 ) << cl.ordering << "\n"; // Ordering
        */
        std::cout << std::left << std::setw( 20 ) << "Cells"
                  << ": " << std::setw( 8 ) << cl.nx << std::setw( 8 ) << cl.ny << std::setw( 8 ) << cl.nz << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Domain"
                  << ": " << std::setw( 8 ) << cl.hx << std::setw( 8 ) << cl.hy << std::setw( 8 ) << cl.hz << "\n"; // Span of Domain
        std::cout << std::left << std::setw( 20 ) << "Periodicity"
                  << ": " << std::setw( 8 ) << // Periodicity
            cl.periodic[0] << std::setw( 8 ) << cl.periodic[1] << std::setw( 8 ) << cl.periodic[2] << "\n";
        std::cout << std::left << std::setw( 20 ) << "Sigma"
                  << ": " << std::setw( 8 ) << cl.sigma << "\n"; // Sigma
        std::cout << std::left << std::setw( 20 ) << "Gravity"
                  << ": " << std::setw( 8 ) << cl.gravity << "\n"; // Gravitational Constant
        std::cout << std::left << std::setw( 20 ) << "Time Steps"
                  << ": " << std::setw( 8 ) << cl.time_steps << "\n"; // Number of Time Steps
        std::cout << std::left << std::setw( 20 ) << "Write Frequency"
                  << ": " << std::setw( 8 ) << cl.write_freq << "\n"; // Time Steps between each Write
        std::cout << "====================================\n";
    }
    timer.writeStop();

    // Call Clamr - Double or Float as Template Arg
    clamr<state_t>( cl, bc, timer );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    // Stop Timer
    timer.overallStop();

    // Print Timer Report
    if ( rank == 0 ) {
        timer.report();
    }

    return 0;
};
