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

// Start Timer Function
void timer_start( std::chrono::high_resolution_clock::time_point* time_start ) {
    * time_start = std::chrono::high_resolution_clock::now();                                                   // Record Start Time
}

// Stop Timer Function
long long int timer_stop( std::chrono::high_resolution_clock::time_point time_start ) {
    auto time_stop = std::chrono::high_resolution_clock::now();                                                 // Record Stop Time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( time_stop - time_start ).count();    // Calculate Elapsed Time in Microseconds

    return duration;
}

}