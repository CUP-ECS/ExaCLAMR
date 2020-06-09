/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#include <Timer.hpp>


namespace Timer{
    
void timer_start( std::chrono::high_resolution_clock::time_point* time_start ) {
    * time_start = std::chrono::high_resolution_clock::now();
}

long long int timer_stop( std::chrono::high_resolution_clock::time_point time_start ) {
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( time_stop - time_start ).count();

    return duration;
}

}