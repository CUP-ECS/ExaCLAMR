/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef INPUT_HPP
#define INPUT_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

#include <getopt.h>
#include <stdlib.h>


// Short Args: n - Cell Count, h - Domain Size, t - Time Steps, w - Write Frequency
static char *shortargs = ( char * )"n::h::t::w::";

// cl_args Struct
template <typename state_t>
struct cl_args {
    int nx, ny, nz, halo_size, time_steps, write_freq;
    state_t hx, hy, hz;
};

// TODO: Usage Message
template <typename state_t>
int parseInput( const int argc, char ** argv, cl_args<state_t>& cl ) {
    cl.nx = 50, cl.ny = 50, cl.nz = 1;          // Default Cell Count

    cl.hx = 50.0, cl.hy = 50.0, cl.hz = 1.0;    // Default Domain Size

    // TODO: Argument for Halo Size and Checks
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
            case 'h':
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
        }
    }

    return 0;
}

#endif