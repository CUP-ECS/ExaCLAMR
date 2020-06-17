/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */
 
#ifndef EXACLAMR_SOLVER_HPP
#define EXACLAMR_SOLVER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

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


template <typename state_t>
class SolverBase {
    public:
        virtual ~SolverBase() = default;
        virtual void solve( const int write_freq, ExaCLAMR::Timer& timer ) = 0;
};


template <class MemorySpace, class ExecutionSpace, typename state_t>
class Solver : public SolverBase<state_t> {
    public:
        /**
         * Constructor
         * 
         * @param
         */
        template <class InitFunc>
        Solver( const cl_args<state_t>& cl, MPI_Comm comm, const InitFunc& create_functor, const Cajita::Partitioner& partitioner, ExaCLAMR::Timer& timer )
        : _halo_size ( cl.halo_size ), _time_steps ( cl.time_steps ), _gravity ( cl.gravity ), _sigma ( cl.sigma ) {
            MPI_Comm_rank( comm, &_rank );

            if ( _rank == 0 && DEBUG ) std::cout << "Created Solver\n";         // DEBUG: Trace Created Solver
                
            // Create Problem Manager
            _pm = std::make_shared<ProblemManager<MemorySpace, ExecutionSpace, state_t>>( cl, partitioner, comm, create_functor );

            // Create Silo Writer
            #ifdef HAVE_SILO
                _silo = std::make_shared<SiloWriter<MemorySpace, ExecutionSpace, state_t>>( _pm );
            #endif

            MPI_Barrier( MPI_COMM_WORLD );

            calcMass( 0 );
        };

        void calcMass( int time_step ) {
            // Get Domain Iteration Space
            auto domain = _pm->mesh()->domainSpace();                                                       // Domain Space to Iterate Over

            // Get State Views
            auto hNew = _pm->get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );               // New Height State View
            auto uNew = _pm->get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );             // New Momentum State View

            state_t summed_height = 0, total_height = 0;

            // Only Loop if Rank is the Specified Rank
            Kokkos::parallel_reduce( Cajita::createExecutionPolicy( domain, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k, state_t& l_height ) {
                l_height += hNew( i, j, k, 0 );
            }, Kokkos::Sum<state_t>( summed_height ) );

            // Get Total Height
            MPI_Allreduce( &summed_height, &total_height, 1, Cajita::MpiTraits<state_t>::type(), MPI_SUM, MPI_COMM_WORLD );

            if ( time_step == 0 ) _initial_mass = total_height;
            else _current_mass = total_height;
        }

        // Print Output of Height Array to Console for Debugging
        void output( const int rank, const int time_step, const state_t current_time, const state_t dt ) {
            // Get Domain Iteration Space
            auto domain = _pm->mesh()->domainSpace();                                                       // Domain Space to Iterate Over

            // Get State Views
            auto hNew = _pm->get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );               // New Height State View
            auto uNew = _pm->get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );             // New Momentum State View

            // Only Loop if Rank is the Specified Rank
            if ( _pm->mesh()->rank() == rank ) {
                for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                    for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                        for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                            if ( DEBUG ) std::cout << std::left << std::setw( 8 ) << hNew( i, j, k, 0 );    // DEBUG: Print Height Array
                        }
                    }
                    if( DEBUG ) std::cout << "\n";                                                          // DEBUG: New Line
                }
            }
        };

        // Solve Routine
        void solve( const int write_freq, ExaCLAMR::Timer& timer ) override {
            if ( _rank == 0 && DEBUG ) std::cout << "Solving!\n";       // DEBUG: Trace Solving

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
                _silo->siloWrite( strdup( "Mesh" ), 0, current_time, mindt );                                // Write State Data
            #endif

            // Loop Over Time
            for (time_step = 1; time_step <= nt; time_step++) {
                timer.computeStart();
                state_t dt = TimeIntegrator::setTimeStep( *_pm, ExecutionSpace(), MemorySpace(), _gravity, _sigma, time_step );       // Calculate Time Step
                timer.computeStop();

                timer.communicationStart();
                // Get Minimum Time Step
                MPI_Allreduce( &dt, &mindt, 1, Cajita::MpiTraits<state_t>::type(), MPI_MIN, MPI_COMM_WORLD );
                timer.communicationStop();

                timer.computeStart();
                TimeIntegrator::step( *_pm, ExecutionSpace(), MemorySpace(), mindt, _gravity, time_step );                          // Perform Calculation
                timer.computeStop();

                timer.communicationStart();
                TimeIntegrator::haloExchange( *_pm, time_step );                                                                    // Perform Communication
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
                        _silo->siloWrite( strdup( "Mesh" ), time_step, current_time, mindt );                                       // Write State Data
                    #endif
                }
                timer.writeStop();
            }
        };
        

    private:
        int _rank, _time_steps, _halo_size;
        state_t _gravity;
        state_t _sigma;
        state_t _initial_mass, _current_mass;
        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, state_t>> _pm;
        #ifdef HAVE_SILO
            std::shared_ptr<SiloWriter<MemorySpace, ExecutionSpace, state_t>> _silo;
        #endif
};

// Create Solver with Templates
template <typename state_t, class InitFunc>
std::shared_ptr<SolverBase<state_t>> createSolver( const cl_args<state_t>& cl,
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
            return std::make_shared<ExaCLAMR::Solver<Kokkos::CudaSpace, Kokkos::Cuda, state_t>>(
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
        return nullptr;
    }
};

}

#endif
