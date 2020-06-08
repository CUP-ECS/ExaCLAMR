#include <Input.hpp>

static char *shortargs = (char *)"n::h::t::w::";

int parseInput( const int argc, char ** argv, cl_args& cl ) {
    cl.nx = 50;
    cl.ny = 50;
    cl.nz = 1;

    cl.hx = 50.0;
    cl.hy = 50.0; 
    cl.hz = 1.0;

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
            case 't':
                cl.tSteps = atoi( optarg );
                break;
            case 'w':
                cl.writeFreq = atoi( optarg );
                break;
        }
    }

    return 0;
}