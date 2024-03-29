
/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Parses optional command line input that gives the user control over problem parameters such as
 * halo cells, number of cells, size of domain, gravitational constant, execution space, number of time steps, 
 * periodicity, dynamic time stepping, and frequency of output.
 */

#ifndef INPUT_HPP
#define INPUT_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>

namespace ExaCLAMR {
    // Short Args: a - Halo Size, b - Mesh Type, d - Domain Size, h - Print Help, g - Gravitational Constant,
    // m - Threading ( Serial or OpenMP or CUDA ), n - Cell Count, o - Ordering, p - Periodicity, s - Sigma, t - Time Steps, w - Write Frequency
    // static char *shortargs = ( char * )"a:b:d:g:hm:n:o:p:s:t:w:";

    // Short Args: a - Halo Size, b - Mesh Type, d - Domain Size, h - Print Help, g - Gravitational Constant,
    // m - Threading ( Serial or OpenMP or CUDA ), n - Cell Count, p - Periodicity, s - Sigma, t - Time Steps, w - Write Frequency
    static char *shortargs = ( char * )"a:b:d:g:h:m:n:p:s:t:w:";

    /**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by command line arguments
 */
    template <typename state_t>
    struct ClArgs {
        int         nx, ny, nz; /**< Number of cells */
        int         halo_size;  /**< Number of halo cells in each direction */
        int         time_steps; /**< Number of time steps in simulation */
        int         write_freq; /**< Write frequency */
        state_t     hx, hy, hz; /**< Size of the domain */
        state_t     gravity;    /**< Gravitation constant */
        state_t     sigma;      /**< Sigma */
        std::string device;     /**< Threading setting ( Serial, OpenMP, CUDA ) */
        std::string meshtype;   /**< Mesh Type ( Regular or AMR ) */
        // std::string ordering;   /**< Ordering Type ( Regular or Hilbert ) */

        std::array<int, 3>     global_num_cells;    /**< Globar array of number of cells */
        std::array<state_t, 6> global_bounding_box; /**< Global bounding box of domain */
        std::array<bool, 3>    periodic;            /**< Periodicity of domain */
    };

    /**
 * Outputs help message explaining command line options.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
    void help( const int rank, char *progname ) {
        if ( rank == 0 ) {
            std::cout << "ExaCLAMR\n";
            std::cout << "Usage: " << progname << "\n";
            std::cout << std::left << std::setw( 10 ) << "-a" << std::setw( 40 ) << "Halo Size (default 2)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-b" << std::setw( 40 ) << "Mesh Type (default Regular)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 ) << "Size of Domain (default 500 500 1)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-h" << std::setw( 40 ) << "Print Help Message" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-g" << std::setw( 40 ) << "Gravitational Constant (default 9.80)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-m" << std::setw( 40 ) << "Thread Setting (default serial)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 ) << "Number of Cells (default 500 500 1)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-p" << std::setw( 40 ) << "Periodicity (default: false false false)" << std::left << "\n";
            std::cout << std::left << std::setw( 20 ) << "  " << std::setw( 50 ) << "-p0 (false false false) -p1 (true false false) -p2(false true false) etc\n";
            std::cout << std::left << std::setw( 10 ) << "-s" << std::setw( 40 ) << "Timestep Sigma Value (default 0.95)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 ) << "Number of Time Steps (default 1000)" << std::left << "\n";
            std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 ) << "Write Frequency (default 100)" << std::left << "\n";
        }
    }

    /**
 * Outputs usage hint if invalid command line arguments are given.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
    void usage( const int rank, char *progname ) {
        /*
        if ( rank == 0 ) std::cout << "usage: " << progname << " [-a halo-size] [-b mesh-type] [-d size-of-domain] [-g gravity] [-h help]"
                                   << " [-m threading] [-n number-of-cells] [-p periodicity] [-s sigma] [-t number-time-steps] [-w write-frequency]\n";
        */
        if ( rank == 0 ) std::cout << "usage: " << progname << " [-a halo-size] [-b mesh-type] [-d size-of-domain] [-g gravity] [-h help]"
                                   << " [-m threading] [-n number-of-cells] [-p periodicity] [-s sigma] [-t number-time-steps] [-w write-frequency]\n";
    }

    /**
 * Parses command line input and updates the command line variables accordingly.\n
 * Usage: ./[program] [-a halo-size] [-b mesh-type] [-d size-of-domain] [-g gravity] [-h help] [-m threading] [-n number-of-cells] [-o ordering] [-p periodicity] [-s sigma] [-t number-time-steps] [-w write-frequency]
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
    template <typename state_t>
    int parseInput( const int rank, const int argc, char **argv, ClArgs<state_t> &cl ) {
        cl.meshtype = "regular"; // Default Mesh Typ

        cl.device = "serial";              // Default Thread Setting
        cl.nx = 500, cl.ny = 500, cl.nz = 1; // Default Cell Count

        cl.hx = 500.0, cl.hy = 500.0, cl.hz = 1.0; // Default Domain Size

        cl.periodic = { false, false, false }; // Default Periodicity

        cl.halo_size  = 2;    // Default Halo Size = 2
        cl.gravity    = 9.80; // Default Gravitational Constant = 9.80
        cl.sigma      = 0.95; // Default Timestep Sigma Value
        cl.time_steps = 1000; // Default Time Steps = 1000
        cl.write_freq = 100;  // Default Write Frequency = 10

        // Initialize
        int c;
	    int  periodicval;

        // Loop through Command-Line Args
        while ( ( c = getopt( argc, argv, shortargs ) ) != -1 ) {
            switch ( c ) {
	    // Halo Size
            case 'a':
                cl.halo_size = atoi( optarg );
                if ( cl.halo_size < 2 ) {
                    if ( rank == 0 ) std::cout << "Halo Size must be greater than or equal to 2\n";
                    return -1;
                }
                break;
            // Mesh Type
            case 'b':
                cl.meshtype = optarg;
                if ( cl.meshtype.compare( "regular" ) && cl.meshtype.compare( "amr" ) ) {
                    if ( rank == 0 ) std::cout << "Valid mesh type inputs are: regular and amr\n";
                    return -1;
                }
                break;
            // Size of Domain
            case 'd':
                cl.hx = atof( optarg );
                cl.hy = atof( optarg );
                cl.hz = 1;
                if ( cl.hz <= 0 ) {
                    if ( rank == 0 ) std::cout << "Extent of domain must be a positive number\n";
                }
                break;
            // Gravitational Constant
            case 'g':
                cl.gravity = atof( optarg );
                if ( cl.gravity < 0.0 ) {
                    if ( rank == 0 ) std::cout << "Gravitational constant must be a positive value\n";
                    return -1;
                }
                break;
            // Help Message
            case 'h':
                help( rank, argv[0] );
                return -1;
            // Threading
            case 'm':
                cl.device = optarg;
                if ( cl.device.compare( "serial" ) && cl.device.compare( "openmp" ) && cl.device.compare( "cuda" ) ) {
                    if ( rank == 0 ) std::cout << "Valid threading inputs are: serial, openmp, and cuda\n";
                    return -1;
                }
                break;
            // Number of Cells
            case 'n':
                cl.nx = atoi( optarg );
                cl.ny = atoi( optarg );
                cl.nz = 1;
                if ( cl.nx <= 1 ) {
                    if ( rank == 0 ) std::cout << "Must be more than 1 cell in each dimension in the mesh\n";
                    return -1;
                }
                break;
            // Periodicity
            case 'p':
                periodicval = atoi( optarg );
                if ( periodicval != 0 ) {
                    if ( rank == 0 ) std::cout << "Periodic boundary conditions are not current enabled\n";
                    return -1;
                } else {
                    for ( int i = 0; i < 3; i++ ) {
                        cl.periodic[i] = ( ( ( periodicval >> i ) & 1 ) == 1 ) ? true : false;
                    }
                }
                break;
            // Timestep Sigma
            case 's':
                cl.sigma = atof( optarg );
                if ( cl.sigma > 2.0 || cl.sigma < 0.0 ) {
                    std::cout << "Sigma must be a value between 0 and 2\n";
                    return -1;
                }
                break;
            // Number of Time Steps
            case 't':
                cl.time_steps = atoi( optarg );
                break;
            // Write Frequency
            case 'w':
                cl.write_freq = atoi( optarg );
                break;
            // Invalid Argument
            case '?':
                usage( rank, argv[0] );
                return -1;
	    }
        }

        // Set Cell Count and Bounding Box Arrays
        cl.global_num_cells    = { cl.nx, cl.ny, cl.nz };
        cl.global_bounding_box = { 0, 0, 0, cl.hx, cl.hy, cl.hz };

        // Return Successfully
        return 0;
    }

} // namespace ExaCLAMR
#endif
