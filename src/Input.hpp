#ifndef INPUT_HPP
#define INPUT_HPP

#define DEBUG 0

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

typedef struct {
    int nx, ny, nz, haloSize, tSteps, writeFreq;
    double hx, hy, hz;
    double rFill;
} cl_args;

int parseInput( const int argc, char** argv, cl_args& cl );

#endif