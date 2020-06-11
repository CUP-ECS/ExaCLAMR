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

#define MICROSECONDS 1.0e-6


namespace ExaCLAMR
{

struct TimerType {
    enum Verbosity {
        OVERALL = 0,
        AGGREGATE = 1
        // FUNCTION = 2,
        // VERBOSE = 3
    };
};

/*
struct TimeVerbose {

};
*/

/*
struct TimeFunction {

};
*/

struct TimeAggregate {
    long long int overall_time;
    long long int setup_time;
    long long int compute_time;
    long long int communication_time;
    long long int write_time;
};


struct TimeOverall {
    long long int overall_time;
};

class Timer{
    using timepoint = std::chrono::high_resolution_clock::time_point;

    public:

        Timer( int verbosity );
        // void setIterations( int iterations );
        void timerStart( timepoint* time_start );             // Start Timer Function
        long long int timerStop( timepoint time_start );      // Stop Timer Function
        void overallStart();
        void overallStop();
        void setupStart();
        void setupStop();
        void computeStart();
        void computeStop();
        void communicationStart();
        void communicationStop();
        void writeStart();
        void writeStop();
        void report();
    
    private:
        int _verbosity;
        
        // struct TimeVerbose _time_verbose;
        // struct TimeFunction _time_function;
        struct TimeAggregate _time_aggregate;
        struct TimeOverall _time_overall;

        timepoint _overall_start;
        timepoint _setup_start;
        timepoint _compute_start;
        timepoint _communication_start;
        timepoint _write_start;
};

}

#endif