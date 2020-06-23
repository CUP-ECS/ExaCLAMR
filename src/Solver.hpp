/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Solver class that stores the problem manager, silo writer, and mesh
 * Iterates over timesteps and calls the Time Integrator to solve and update state arrays
 * Writes output to Silo files on specified time steps
 */
 
#ifndef EXACLAMR_SOLVER_HPP
#define EXACLAMR_SOLVER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

// Include Statements
#include <ExaCLAMR.hpp>
#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <TimeIntegration.hpp>
#include <Timer.hpp>

#ifdef HAVE_SILO
    #include <SiloWriter.hpp>
#endif

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <memory>

// Seconds to Microseconds
#define MICROSECONDS 1.0e-6


namespace ExaCLAMR
{

/**
 * The SolverBase Class
 * @class SolverBase
 * @brief SolverBase class to provide virutal functions for actual Solver class
 **/
template <typename state_t>
class SolverBase {
    public:

        /**
         * Destructor
         **/
        virtual ~SolverBase() = default;

        /**
         * Solves PDEs on a regular grid
         * @param write_freq Frequency of writing output and results
         * @param timer Timer used to profile performance
         **/
        virtual void solve( const int write_freq, ExaCLAMR::Timer& timer ) = 0;
};


/**
 * The Solver Class
 * @class Solver
 * @brief Solver class to store problem manager and silo writer and to iterate over specified time steps and write results to file
 **/
template <class MemorySpace, class ExecutionSpace, typename state_t>
class Solver : public SolverBase<state_t> {
    public:
        /**
         * Constructor
         * Determine rank
         * Create new problem manager object
         * Create new silo object if silo is available
         * Calculate initial mass of the system
         * Set private variables, halo size, time steps, gravity, and sigma
         * 
         * @param cl Command line arguments
         * @param comm MPI communicator
         * @param create_functor Initialization function
         * @param partitioner Cajita MPI Partitioner
         * @param timer ExaCLAMR timer to profile performance
         */
        template <class InitFunc>
        Solver( const ExaCLAMR::ClArgs<state_t>& cl, MPI_Comm comm, const InitFunc& create_functor, const Cajita::Partitioner& partitioner, ExaCLAMR::Timer& timer )
        : _halo_size ( cl.halo_size ), _time_steps ( cl.time_steps ), _gravity ( cl.gravity ), _sigma ( cl.sigma ) {
            MPI_Comm_rank( comm, &_rank );
            // DEBUG: Trace Created Solver
            if ( _rank == 0 && DEBUG ) std::cout << "Created Solver\n";
                
            // Create Problem Manager
            _pm = std::make_shared<ProblemManager<MemorySpace, ExecutionSpace, state_t>>( cl, partitioner, comm, create_functor );

            // Create Silo Writer
            #ifdef HAVE_SILO
                _silo = std::make_shared<SiloWriter<MemorySpace, ExecutionSpace, state_t>>( _pm );
            #endif

            MPI_Barrier( MPI_COMM_WORLD );

            calcMass( 0 );
        };

        /**
         * Calculates the current mass of the system and stores in either _initial_mass or _current_mass
         * @param time_step Current time step
         **/
        void calcMass( int time_step ) {
            // Get Domain Iteration Space
            auto domain = _pm->mesh()->domainSpace();

            // Get State Views
            auto hNew = _pm->get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );
            auto uNew = _pm->get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );

            state_t summed_height = 0, total_height = 0;

            // Only Loop if Rank is the Specified Rank
            Kokkos::parallel_reduce( Cajita::createExecutionPolicy( domain, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k, state_t& l_height ) {
                l_height += hNew( i, j, k, 0 );
            }, Kokkos::Sum<state_t>( summed_height ) );

            // Get Total Height
            MPI_Allreduce( &summed_height, &total_height, 1, Cajita::MpiTraits<state_t>::type(), MPI_SUM, MPI_COMM_WORLD );

            if ( time_step == 0 ) _initial_mass = total_height;
            else _current_mass = total_height;
        };

        /**
         * Print Output of Height Array to Console for Debugging
         * @param rank Rank to print output
         * @param time_step Current time step
         * @param current_time Current simulation time
         * @param dt Time step (dt)
         **/
        void output( const int rank, const int time_step, const state_t current_time, const state_t dt ) {
            // Get Domain Iteration Space
            auto domain = _pm->mesh()->domainSpace();

            // Get State Views
            auto hNew = _pm->get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );
            auto uNew = _pm->get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );

            // Only Loop if Rank is the Specified Rank
            if ( _pm->mesh()->rank() == rank ) {
                for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                    for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                        for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                            // DEBUG: Print Height array
                            if ( DEBUG ) std::cout << std::left << std::setw( 8 ) << hNew( i, j, k, 0 );
                        }
                    }
                    // DEBUG: New Line
                    if( DEBUG ) std::cout << "\n";
                }
            }
        };


        /**
         * Solves PDEs on a regular grid
         * @param write_freq Frequency of writing output and results
         * @param timer Timer used to profile performance
         **/
        void solve( const int write_freq, ExaCLAMR::Timer& timer ) override {
            // DEBUG: Trace Solving
            if ( _rank == 0 && DEBUG ) std::cout << "Solving!\n";

            int time_step = 0;
            int nt = _time_steps;
            state_t current_time = 0.0, mindt = 0.0;

            // Rank 0 Prints Initial Iteration and Time
            if (_rank == 0 ) {
                // Print Iteration and Current Time
                std::cout << std::left << std::setw( 12 ) << "Iteration: " << std::left << std::setw( 12 ) << 0 <<
                std::left << std::setw( 15 ) << "Current Time: " << std::left << std::setw( 12 ) << current_time << 
                std::left << std::setw( 15 ) << "Total Mass: " << std::left << std::setw( 12 ) << _initial_mass << "\n"; 

                // DEBUG: Call Output Routine                  
                if ( DEBUG ) output( 0, time_step, current_time, mindt );                                   
            }

            // Write Initial Data to File with Silo
            #ifdef HAVE_SILO
                _silo->siloWrite( strdup( "Mesh" ), 0, current_time, mindt );
            #endif

            // Loop Over Time
            for (time_step = 1; time_step <= nt; time_step++) {
                timer.computeStart();
                // Calculate Time Step
                state_t dt = TimeIntegrator::setTimeStep( *_pm, ExecutionSpace(), MemorySpace(), _gravity, _sigma, time_step );
                timer.computeStop();

                timer.communicationStart();
                // Get Minimum Time Step
                MPI_Allreduce( &dt, &mindt, 1, Cajita::MpiTraits<state_t>::type(), MPI_MIN, MPI_COMM_WORLD );
                timer.communicationStop();

                timer.computeStart();
                // Perform Calculation
                TimeIntegrator::step( *_pm, ExecutionSpace(), MemorySpace(), mindt, _gravity, time_step );
                timer.computeStop();

                timer.communicationStart();
                // Halo Exchange
                TimeIntegrator::haloExchange( *_pm, ExecutionSpace(), MemorySpace(), mindt, time_step );
                timer.communicationStop();

                timer.computeStart();
                calcMass( time_step );
                state_t mass_change = _initial_mass - _current_mass;
                timer.computeStop();

                // Increment Current Time
                current_time += mindt;

                // Output and Write File every Write Frequency Time Steps
                timer.writeStart();
                if ( 0 == time_step % write_freq ) {
                    if ( 0 == _rank ) std::cout << std::left << std::setw( 12 ) << "Iteration: " << std::left << std::setw( 12 ) << time_step <<
                    std::left << std::setw( 15 ) << "Current Time: " << std::left << std::setw( 12 ) << current_time << 
                    std::left << std::setw( 15 ) << "Mass Change: " << std::left << std::setw( 12 ) << mass_change << "\n";

                    // DEBUG: Call Output Routine
                    if ( DEBUG ) output( 0, time_step, current_time, mindt );

                    // Write Current State Data to File with Silo
                    #ifdef HAVE_SILO
                        _silo->siloWrite( strdup( "Mesh" ), time_step, current_time, mindt );
                    #endif
                }
                timer.writeStop();
            }
        };
        

    private:
        int _rank;                                                                          /**< Rank of solver */
        int _time_steps;                                                                    /**< Number of time steps to solve for */
        int _halo_size;                                                                     /**< Halo size of the mesh */

        state_t _gravity;                                                                   /**< Gravitational constant */
        state_t _sigma;                                                                     /**< Sigma used to control CFL number and calculate time step */
        state_t _initial_mass;                                                              /**< Initial mass of the system */
        state_t _current_mass;                                                              /**< Current mass of the system */

        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, state_t>> _pm;          /**< Problem Manager object */
        #ifdef HAVE_SILO
            std::shared_ptr<SiloWriter<MemorySpace, ExecutionSpace, state_t>> _silo;        /**< Silo writer object */
        #endif
};


/**
 * Create Solver Pointer with Templates based on specified ExecutionSpace and MemorySpace
 * @param cl Command line arguments
 * @param comm MPI communicator
 * @param create_functor Initialization function
 * @param partitioner Cajita MPI Partitioner
 * @param timer ExaCLAMR timer to profile performance
**/
template <typename state_t, class InitFunc>
std::shared_ptr<SolverBase<state_t>> createSolver( const ExaCLAMR::ClArgs<state_t>& cl,
                                            MPI_Comm comm, 
                                            const InitFunc& create_functor,
                                            const Cajita::Partitioner& partitioner,
                                            ExaCLAMR::Timer& timer ) {
    // Serial
    if ( 0 == cl.device.compare( "serial" ) ) {
        #ifdef KOKKOS_ENABLE_SERIAL
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::Serial, state_t>>(
                cl,
                comm,
                create_functor,
                partitioner,
                timer );
        #else
            throw std::runtime_error( "Serial Backend Not Enabled" );
        #endif
    }
    // OpenMP
    else if ( 0 == cl.device.compare( "openmp" ) ) {
        #ifdef KOKKOS_ENABLE_OPENMP
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::OpenMP, state_t>>(
                cl,
                comm, 
                create_functor,
                partitioner,
                timer );
        #else
            throw std::runtime_error( "OpenMP Backend Not Enabled" );
        #endif
    }
    // Cuda
    else if ( 0 == cl.device.compare( "cuda" ) ) {
        #ifdef KOKKOS_ENABLE_CUDA
            return std::make_shared<ExaCLAMR::Solver<Kokkos::CudaUVMSpace, Kokkos::Cuda, state_t>>(
                cl,
                comm,
                create_functor,
                partitioner,
                timer );
        #else
            throw std::runtime_error( "Cuda Backend Not Enabled" );
        #endif
    }
    // Otherwise
    else {
        throw std::runtime_error( "Invalid Backend" );
    }
};

}

#endif
