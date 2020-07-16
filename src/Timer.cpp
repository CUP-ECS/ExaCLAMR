/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * ExaCLAMR timer to use for profiling of the program
 */

#include <Timer.hpp>

#include <iostream>

namespace ExaCLAMR {

    Timer::Timer( int verbosity )
        : _verbosity( verbosity ) {
        if ( _verbosity == TimerType::OVERALL ) _time_overall.overall_time = 0;
        if ( _verbosity == TimerType::AGGREGATE ) {
            _time_aggregate.overall_time       = 0;
            _time_aggregate.setup_time         = 0;
            _time_aggregate.compute_time       = 0;
            _time_aggregate.communication_time = 0;
            _time_aggregate.write_time         = 0;
        }
    }

    /*
void Timer::setIterations( int iterations ) {
    if ( _verbosity == TimerType::VERBOSE ) struct TimeVerbose *_time_log = ( struct TimeVerbose* )calloc( iterations, sizeof( struct TimeVerbose ) );
}
*/

    // Start Timer Method
    void Timer::timerStart( timepoint *time_start ) {
        // Record Start Time
        *time_start = std::chrono::high_resolution_clock::now();
    }

    // Stop Timer Method
    long long int Timer::timerStop( timepoint time_start ) {
        // Record Stop Time
        auto time_stop = std::chrono::high_resolution_clock::now();
        // Calculate Elapsed Time in Microseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( time_stop - time_start ).count();

        return duration;
    }

    // Overall Start Method
    void Timer::overallStart() {
        timerStart( &_overall_start );
    }

    // Overall Stop Method
    void Timer::overallStop() {
        long long int duration = timerStop( _overall_start );
        _time_overall.overall_time += duration;

        if ( _verbosity == TimerType::AGGREGATE ) _time_aggregate.overall_time += duration;
    }

    // Start Setup Timer Method
    void Timer::setupStart() {
        timerStart( &_setup_start );
    }

    // Stop Setup Timer Method
    void Timer::setupStop() {
        long long int duration = timerStop( _setup_start );
        if ( _verbosity == TimerType::AGGREGATE ) _time_aggregate.setup_time += duration;
    }

    // Start Compute Timer Method
    void Timer::computeStart() {
        timerStart( &_compute_start );
    }

    // Stop Compute Timer Method
    void Timer::computeStop() {
        long long int duration = timerStop( _compute_start );

        if ( _verbosity == TimerType::AGGREGATE ) _time_aggregate.compute_time += duration;
    }

    // Start Communicate Timer Method
    void Timer::communicationStart() {
        timerStart( &_communication_start );
    }

    // Stop Communicate Timer Method
    void Timer::communicationStop() {
        long long int duration = timerStop( _communication_start );

        if ( _verbosity == TimerType::AGGREGATE ) _time_aggregate.communication_time += duration;
    }

    // Start Write Timer Method
    void Timer::writeStart() {
        timerStart( &_write_start );
    }

    // Stop Write Timer Method
    void Timer::writeStop() {
        long long int duration = timerStop( _write_start );

        if ( _verbosity == TimerType::AGGREGATE ) _time_aggregate.write_time += duration;
    }

    // Timer Report Method
    void Timer::report() {
        std::cout << "Timing Report\n";

        // If Timer Type is Overall
        if ( _verbosity == TimerType::OVERALL ) {
            std::cout << "Overall Wall Time: " << _time_overall.overall_time << "\n";
        }

        // If Timer Type is Aggregate
        else if ( _verbosity == TimerType::AGGREGATE ) {
            std::cout << "Overall Wall Time: " << _time_aggregate.overall_time * MICROSECONDS << " seconds\n";
            std::cout << "Total Setup Time: " << _time_aggregate.setup_time * MICROSECONDS << " seconds\n";
            std::cout << "Total Compute Time: " << _time_aggregate.compute_time * MICROSECONDS << " seconds\n";
            std::cout << "Total Communication Time: " << _time_aggregate.communication_time * MICROSECONDS << " seconds\n";
            std::cout << "Total Output Writing Time: " << _time_aggregate.write_time * MICROSECONDS << " seconds\n";
        }

        // else if ( _verbosity == TimerType::FUNCTION ) {}
        // else if ( _verbosity == TimerType::VERBOSE ) {}
    }

} // namespace ExaCLAMR