  
/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Parses optional command line input that gives the user control over problem parameters such as
 * number of cells, size of domain, number of time steps, and frequency of output.
 */

#ifndef INPUT_HPP
#define INPUT_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

#include <iostream>
#include <iomanip>
#include <getopt.h>
#include <stdlib.h>


// Short Args: n - Cell Count, d - Domain Size, t - Time Steps, w - Write Frequency, a - Halo Size, h - Print Help
static char *shortargs = ( char * )"n::d::t::w::a::h";

/**
 * Template struct to organize and keep track of parameters controlled by command line arguments
 */
template <typename state_t>
struct cl_args {
    int nx, ny, nz;         /**< Number of cells */
    int halo_size;          /**< Number of halo cells in each direction */
    int time_steps;         /**< Number of time steps in simulation */
    int write_freq;         /**< Write frequency */
    state_t hx, hy, hz;     /**< Size of the domain */
};


/**
 * Outputs help message explaining command line options.
 */
void help( const int rank, char* progname ){
    if ( rank == 0 ) {
        std::cout << "ExaCLAMR\n";
        std::cout << "Usage: " << progname << "\n";
        std::cout << std::left << std::setw( 10 ) << "-a" << std::setw( 40 ) << "Halo Size (default 2)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 ) << "Size of Domain (default 50 50 1)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 ) << "Number of Cells (default 50 50 1)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 ) << "Number of Time Steps (default 3000)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 ) << "Write Frequency (default 100)" << std::left << "\n";
    }
}


/**
 * Outputs usage hint if invalid command line arguments are given.
 */
void usage( const int rank, char* progname ) {
    if ( rank == 0 ) std::cout << "usage: " << progname << " [-n number-of-cells] [-d size-of-domain] [-t number-time-steps] [-w write-frequency] [-a halo-size]\n";
}


/**
 * Parses command line input and updates the command line variables accordingly.\n
 * Usage: ./[program] [-n number-of-cells] [-d size-of-domain] [-t number-time-steps] [-w write-frequency] [-a halo-size]
 */
template <typename state_t>
int parseInput( const int rank, const int argc, char ** argv, cl_args<state_t>& cl ) {
    cl.nx = 50, cl.ny = 50, cl.nz = 1;          // Default Cell Count

    cl.hx = 50.0, cl.hy = 50.0, cl.hz = 1.0;    // Default Domain Size

    cl.halo_size = 2;                           // Default Halo Size = 2
    cl.time_steps = 3000;                       // Default Time Steps = 3000
    cl.write_freq = 100;                        // Default Write Frequency = 10

    // Initialize
    char c;
    int optindex;

    // Loop through Command-Line Args
    // TODO: Value Checks and Error Handling
    while ( ( c = getopt( argc, argv, shortargs ) ) != -1 ) {
        switch ( c ) {
            // Number of Cells
            // TODO: Make Nx, Ny Independent
            case 'n':
                cl.nx = atoi( optarg );
                cl.ny = atoi( optarg );
                cl.nz = 1;
                break;
            // Size of Domain
            // TODO: Make Hx, Hy Independent
            case 'd':
                cl.hx = atof( optarg );
                cl.hy = atof( optarg );
                cl.hz = 1;
                break;
            // Number of Time Steps
            case 't':
                cl.time_steps = atoi( optarg );
                break;
            // Write Frequency
            case 'w':
                cl.write_freq = atoi( optarg );
                break;
            // Halo Size
            case 'a':
                cl.halo_size = atoi ( optarg );
                if ( cl.halo_size < 2 ) {
                    if ( rank == 0 ) std::cout << "Halo Size must be greater than or equal to 2\n";
                    return -1;
                }
                break;
            // Help Message
            case 'h':
                help( rank, argv[0] );
                return -1;
            // Invalid Argument
            case '?':
                usage( rank, argv[0] );
                return -1;
        }
    }

    return 0;
}

#endif