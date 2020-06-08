#include <Input.hpp>

static char *shortargs = (char *)"n::h::r::t::w::";

int parseInput( const int argc, char ** argv, cl_args& cl ) {
    cl.nx = 50;
    cl.ny = 50;
    cl.nz = 1;

    cl.hx = 50.0;
    cl.hy = 50.0; 
    cl.hz = 1.0;

    cl.rFill = 6.0;

    cl.haloSize = 2;
    cl.tSteps = 3000;
    cl.writeFreq = 100;

    char c;
    int optindex;

    while ( ( c = getopt( argc, argv, shortargs ) ) != -1 ) {
        switch ( c ) {
            case 'n':
                cl.nx = atoi( optarg );
                cl.ny = atoi( optarg );
                cl.nz = 1;
                break;
            case 'h':
                cl.hx = atof( optarg );
                cl.hy = atof( optarg );
                cl.hz = 1;
                break;
            case 'r':
                cl.rFill = atof( optarg );
                break;
            case 't':
                cl.tSteps = atoi( optarg );
                break;
            case 'w':
                cl.writeFreq = atoi( optarg );
                break;
        }
    }

        // Print Results
    printf( "ExaCLAMR\n" );
    printf( "=======Command line arguments=======\n" );
    printf( "%-25s%- 8d%- 8d%- 8d\n", "Cells:", cl.nx, cl.ny, cl.nz );
    printf( "%-25s%- 4.4f%- 4.4f%- 4.4f\n", "Domain:", cl.hx, cl.hy, cl.hz );
    printf( "%-25s%- 4.4f\n", "Fill Radius:", cl.rFill );
    printf( "%-25s%- 8d\n", "Time Steps:", cl.tSteps );
    printf( "%-25s%- 8d\n", "Write Frequency:", cl.writeFreq );
    printf("====================================\n");

    return 0;
}