/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 *
 * @version 0.1.0
 * 
 * @section LICENSE
 * 
 * @section DESCRIPTION
 * 
 */
 
#ifndef EXACLAMR_SOLVER_HPP
#define EXACLAMR_SOLVER_HPP

#define DEBUG 0

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <TimeIntegration.hpp>
#include <SiloWriter.hpp>

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <memory>

namespace ExaCLAMR
{


class SolverBase
{
    public:
        virtual ~SolverBase() = default;
        virtual void solve( const int write_freq ) = 0;
};

template<class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
    public:

        /**
         * Constructor
         * 
         * @param
         */
        template<class InitFunc>
        Solver( MPI_Comm comm, 
                const InitFunc& create_functor,
                const std::array<double, 6>& global_bounding_box, 
                const std::array<int, 3>& global_num_cell, 
                const std::array<bool, 3>& periodic,
                const Cajita::Partitioner& partitioner,
                const int halo_size, 
                const double t_steps, 
                const double gravity )
        : _halo_size ( halo_size ), _tsteps ( t_steps ), _gravity ( gravity )
        {
            MPI_Comm_rank( comm, &_rank );

            if ( _rank == 0 && DEBUG ) std::cout << "Created Solver\n";

            _mesh = std::make_shared<Mesh<MemorySpace, ExecutionSpace>> ( global_bounding_box, 
                                            global_num_cell, 
                                            periodic, 
                                            partitioner, 
                                            halo_size, 
                                            comm,
                                            ExecutionSpace()
                                            );
                
            _pm = std::make_shared<ProblemManager<MemorySpace, ExecutionSpace, double>>( _mesh, create_functor, ExecutionSpace() );

            _silo = std::make_shared<SiloWriter<MemorySpace, ExecutionSpace>>( _pm );

            MPI_Barrier( MPI_COMM_WORLD );
        };

        void output( const int rank, const int tstep, const double current_time, const double dt ) {
            int a, b;
            if ( tstep % 2 == 0 ) {
                a = 0;
                b = 1;
            }
            else {
                a = 1;
                b = 0;
            }

            auto owned_cells = _pm->mesh()->localGrid()->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
            auto domain = _pm->mesh()->domainSpace();

            auto uNew = _pm->get( Location::Cell(), Field::Velocity(), b );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), b );

            double summedHeight = 0;
            if ( _pm->mesh()->rank() == rank ) {
                for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                    for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                        for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                            if ( DEBUG ) std::cout << std::left << std::setw(8) << hNew( i, j, k, 0 );
                            summedHeight += hNew( i, j, k, 0 );
                        }
                    }
                    if( DEBUG ) std::cout << "\n";
                }

                // Proxy Mass Conservation
                if ( DEBUG ) std::cout << "Summed Height: " << summedHeight << "\n";
            }
        };

        void solve( const int write_freq ) override {
            if ( _rank == 0 && DEBUG ) std::cout << "Solving!\n";

            int nt = _tsteps;
            double current_time = 0.0;
            int a, b;

            if (_rank == 0 ) {
                std::cout << std::left << std::setw(12) << "Iteration: " << 0 << std::left << std::setw(15) << "\tCurrent Time: " << current_time << "\n";
                if ( DEBUG ) output( 0, 0, 0, 0 );
                #ifdef HAVE_SILO
                    _silo->siloWrite( strdup( "Mesh" ), 0, current_time, 0, 1 );
                #endif

            }

            for (int t = 1; t <= nt; t++) {
                if ( t % 2 == 0 ) {
                    a = 0;
                    b = 1;
                }
                else {
                    a = 1;
                    b = 0;
                }

                double mindt;
                double dt = TimeIntegrator::setTimeStep( *_pm, ExecutionSpace(), MemorySpace(), _gravity, 0.95, 0, 1 );

                MPI_Allreduce( &dt, &mindt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

                TimeIntegrator::step( *_pm, ExecutionSpace(), MemorySpace(), mindt, _gravity, a, b );
                current_time += mindt;

                if ( 0 == t % write_freq ) {
                    if ( 0 == _rank ) std::cout << std::left << std::setw(12) << "Iteration: " << std::setw(5) << t << std::left << std::setw(15) << "\tCurrent Time: " << current_time << "\n";

                    if ( DEBUG ) output( 0, t, current_time, mindt );

                    #ifdef HAVE_SILO
                        _silo->siloWrite( strdup( "Mesh" ), t, current_time, mindt, b );
                    #endif
                }
            }
        };

    private:

        double _tsteps;
        double _gravity;
        int _halo_size;
        int _rank;
        std::shared_ptr<Mesh<MemorySpace, ExecutionSpace>> _mesh;
        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, double>> _pm;
        std::shared_ptr<SiloWriter<MemorySpace, ExecutionSpace>> _silo;
};

template<class InitFunc>
std::shared_ptr<SolverBase> createSolver( const std::string& device,
                                            MPI_Comm comm, 
                                            const InitFunc& create_functor,
                                            const std::array<double, 6>& global_bounding_box, 
                                            const std::array<int, 3>& global_num_cell,
                                            const std::array<bool, 3>& periodic,
                                            const Cajita::Partitioner& partitioner, 
                                            const int halo_size, 
                                            const double t_steps, 
                                            const double gravity ) 
{
    if ( 0 == device.compare( "serial" ) ){
        #ifdef KOKKOS_ENABLE_SERIAL
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::Serial>>(
                comm, 
                create_functor,
                global_bounding_box, 
                global_num_cell, 
                periodic,
                partitioner,
                halo_size, 
                t_steps, 
                gravity );
        #else
            throw std::runtime_error( "Serial Backend Not Enabled" );
        #endif
    }
    else if ( 0 == device.compare( "openmp" ) ) {
        #ifdef KOKKOS_ENABLE_OPENMP
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::OpenMP>>(
                comm, 
                create_functor,
                global_bounding_box, 
                global_num_cell, 
                periodic,
                partitioner,
                halo_size, 
                t_steps, 
                gravity );
        #else
            throw std::runtime_error( "OpenMP Backend Not Enabled" );
        #endif
    }
    else {
        throw std::runtime_error( "Invalid Backend" );
        return nullptr;
    }
};

}
#endif
