/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef TIMER_HPP
#define TIMER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

#include <chrono>


namespace Timer
{

void timer_start( std::chrono::high_resolution_clock::time_point* time_start );             // Start Timer Function
long long int timer_stop( std::chrono::high_resolution_clock::time_point time_start );      // Stop Timer Function

}

#endif