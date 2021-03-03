/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * ExaCLAMR timer to use for profiling of the program
 */

#ifndef TIMER_HPP
#define TIMER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <chrono>

// Microsecond to second conversion
#define MICROSECONDS 1.0e-6

namespace ExaCLAMR {

    /**
 * @struct TimerType
 * @brief Template struct to keep enum of the Timer type 
 **/
    struct TimerType {
        enum Verbosity {
            OVERALL   = 0,
            AGGREGATE = 1
            // FUNCTION = 2,
            // VERBOSE = 3
        };
    };

    /*
struct TimeVerbose {};
*/

    /*
struct TimeFunction {};
*/

    /**
 * @struct TimeAggregate
 * @brief Template struct for Timer of type "aggregate"
 **/
    struct TimeAggregate {
        long long int overall_time;       /**< Overall time tracker */
        long long int setup_time;         /**< Setup time tracker */
        long long int compute_time;       /**< Compute time tracker */
        long long int communication_time; /**< Communication time tracker */
        long long int write_time;         /**< Write/Output time tracker */
    };

    /**
 * @struct TimeOverall
 * @brief Template struct for Timer of type "overall"
 */
    struct TimeOverall {
        long long int overall_time; /**< Overall time tracker */
    };

    /**
 * The Timer Class
 * @class Timer
 * @brief Timer class to profile the ExaCLAMR solver
 **/
    class Timer {
        using timepoint = std::chrono::high_resolution_clock::time_point;

      public:
        /**
         * Constructor
         * Creates a new timer with fields to track various events.
         * Initializes all timer structs and time stamps to 0.
         * @param verbosity Indicates the granularity we wish to profile to program at (Overall, Aggregate, Function, Verbose)
         */
        Timer( int verbosity );

        // void setIterations( int iterations );

        /**
         * Start timer and store time stamp of starting time - generalization used by internal methods
         * @param time_start Pointer to time stamp of starting time
         **/
        void timerStart( timepoint *time_start );

        /**
         * Stop timer and calculates the elapsed time - generalization used by internal methods
         * @param time_start Time stamp of starting time used to calcluate elapsed time
         * @return Duration in microseconds
         **/
        long long int timerStop( timepoint time_start );

        /**
         * Start overall time tracker on Overall, Aggregate, Function, and Verbose timer levels
         **/
        void overallStart();

        /**
         * Stop overall time tracker on Overall, Aggregate, Function, and Verbose timer levels
         **/
        void overallStop();

        /**
         * Start setup time tracker on Aggregate timer level
         **/
        void setupStart();

        /**
         * Stop setup time tracker on Aggregate timer level
         **/
        void setupStop();

        /**
         * Start compute time tracker on Aggregate timer level
         **/
        void computeStart();

        /**
         * Stop compute time tracker on Aggregate timer level
         **/
        void computeStop();

        /**
         * Start communication time tracker on Aggregate timer level
         **/
        void communicationStart();

        /**
         * Stop communication time tracker on Aggregate timer level
         **/
        void communicationStop();

        /**
         * Start write time tracker on Aggregate timer level
         **/
        void writeStart();

        /**
         * Stop write time tracker on Aggregate timer level
         **/
        void writeStop();

        /**
         * Print out timing report
         **/
        void report();

      private:
        int _verbosity; /**< Verbosity indicator */

        // struct TimeVerbose _time_verbose;
        // struct TimeFunction _time_function;
        struct TimeAggregate _time_aggregate; /**< Agggregate time tracker struct */
        struct TimeOverall   _time_overall;   /**< Overall time tracker struct */

        timepoint _overall_start;       /**< Overall start time stamp */
        timepoint _setup_start;         /**< Setup start time stamp */
        timepoint _compute_start;       /**< Compute start time stamp */
        timepoint _communication_start; /**< Communication start time stamp */
        timepoint _write_start;         /**< Write start time stamp */
    };

} // namespace ExaCLAMR

#endif
